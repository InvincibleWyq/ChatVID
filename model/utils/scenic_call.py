import functools

from absl import app
from absl import flags
from absl import logging

from clu import metric_writers
from clu import platform
import flax.linen as nn
import jax
from ml_collections import config_flags
import tensorflow as tf

import sys, os
from pathlib import Path
# append current path to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent / "scenic"))

import logging
import flax
from flax import jax_utils
from flax.training import checkpoints
from scenic.projects.vid2seq import models, trainer
from scenic.train_lib_deprecated import train_utils
from scenic import app
import ml_collections
import numpy as np
import jax.numpy as jnp
from clu import metric_writers
from scenic.projects.vid2seq.datasets.dense_video_captioning_tfrecord_dataset import get_datasets
from scenic.projects.vid2seq import dvc_eval

MAX_CAPTION_STR_LEN = 200
MAX_KEY_STR_LEN = 400

class ScenicModel:
    def __init__(self, flags):
        self.FLAGS = flags
        jax.config.config_with_absl()
        run = (functools.partial(self._run_main, main=self._init_model))
        run(self._init_model)
    def _run_main(self, argv, *, main):
        """Runs the `main` method after some initial setup."""
        del argv
        # Hide any GPUs form TensorFlow. Otherwise, TF might reserve memory and make
        # it unavailable to JAX.
        tf.config.experimental.set_visible_devices([], 'GPU')

        # Enable wrapping of all module calls in a named_call for easier profiling:
        nn.enable_named_call()

        logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
        logging.info('JAX devices: %r', jax.devices())

        # Add a note so that we can tell which task is which JAX host.
        # (task 0 is not guaranteed to be the host 0)
        platform.work_unit().set_task_status(
            f'host_id: {jax.process_index()}, host_count: {jax.process_count()}')
        if jax.process_index() == 0:
            platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                                self.FLAGS.workdir, 'Workdir')
        self.FLAGS.config.dataset_configs.base_dir = self.FLAGS.data_dir
        rng = jax.random.PRNGKey(self.FLAGS.config.rng_seed)
        logging.info('RNG: %s', rng)

        writer = metric_writers.create_default_writer(
            self.FLAGS.workdir, just_logging=jax.process_index() > 0, asynchronous=True)

        return main(rng=rng, config=self.FLAGS.config, workdir=self.FLAGS.workdir, writer=writer)


    def _init_model(self, rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
        data_rng, rng = jax.random.split(rng)
        dataset_dict = get_datasets(config, data_rng=data_rng)

        datasets_metadata = {
            name: ds.meta_data
            for name, ds in dataset_dict.items()
        }
        all_datasets = []
        all_datasets_num_train_examples = []
        for name, metadata in datasets_metadata.items():
            all_datasets.append(name)
            all_datasets_num_train_examples.append(
                metadata.get('num_train_examples', 0))
        dataset = dataset_dict[all_datasets[0]]

        model_cls = models.DenseVideoCaptioningModel
        model = model_cls(config, dataset.meta_data)
        train_state, start_step = trainer.init_state(model, dataset, config,
                                                    workdir, rng)

        self.train_state = jax_utils.replicate(train_state)
        logging.info('Number of processes is %s', jax.process_count())
        del rng

        import functools
        self.infer_step_pmapped = jax.pmap(
        functools.partial(
            trainer.infer_step,
            model=model,
            config=config,
            debug=config.debug_eval),
        axis_name='batch',
        )
        
        self.tokenizer = trainer.get_tokenizer(config)
        # dsname = 'validation'
        # self.iterator = dataset.valid_iter[dsname]
        
        self.config = config
        self.data_rng = data_rng
    
    def __call__(self, data_dir=None):
        # self.FLAGS.config.dataset_configs.base_dir = data_dir
        dataset_dict = get_datasets(self.config, data_rng=self.data_rng)
        self.iterator = dataset_dict["youcook"].valid_iter['validation']
        batch = next(self.iterator)
        
        train_state = train_utils.sync_model_state_across_replicas(self.train_state)
        eval_packs = {}
        keys = []
        eval_pack = {
        'gts':
            dvc_eval.convert_strings_to_uint8_arrays(
                batch['caption_strings'], MAX_CAPTION_STR_LEN),
        'key':
            dvc_eval.convert_strings_to_uint8_arrays(
                batch['videoid'], MAX_KEY_STR_LEN),
        'batch_mask':
            batch['batch_mask'],
        'duration':
            batch['duration'],
        'gts_start':
            batch['timestamp_start'],
        'gts_end':
            batch['timestamp_end'],
        'split':
            batch['split'] if 'split' in batch else
            np.ones_like(batch['timestamp_start']),
        }
        to_del = ['caption_strings', 'key', 'videoid', 'timestamp_start',
                    'timestamp_end', 'split']  # 'duration',
        for x in to_del:
            if x in batch:
                del batch[x]
        
        # import pdb
        # pdb.set_trace()
        
        _, preds = self.infer_step_pmapped(train_state, batch) #model, config)
        # import pdb
        # pdb.set_trace()
        eval_pack['pred'] = preds
        eval_pack = jax.tree_map(
            lambda x: x.reshape((np.prod(x.shape[:2]),) + x.shape[2:]), eval_pack)
        
        vocabulary_size = self.config.dataset_configs.vocabulary_size
        # pred_text = trainer.decode_tokens(preds, tokenizer, vocabulary_size)

        # print(preds, pred_text)
        format_outputs = []
        for i, valid in enumerate(eval_pack['batch_mask']):
            print("===============video[", str(0), "]====================")
            if valid:
                key = dvc_eval.convert_uint8_array_to_string(eval_pack['key'][i])
            if key in eval_packs:  # redundant video
                continue
            keys.append(key)

            pred, pred_timestamps = [], []
            # get indexes in the predicted seq that delimit the pred segments
            indexes = [
                j for j in range(len(eval_pack['pred'][i]) - 1)
                if eval_pack['pred'][i][j] >= vocabulary_size and
                eval_pack['pred'][i][j + 1] >= vocabulary_size
            ]  # pylint: disable=g-complex-comprehension

            last_processed = -2
            order = self.config.dataset_configs.order

            # iterate over predicted segments and decode them
            for j in range(len(indexes)):
                if indexes[j] == last_processed + 1:  # 3 timestamps != 2 events
                    continue
                
                # get predicted tokens and transform to string
                if order == 'ld':
                    start_idx = indexes[j] + 2
                    end_idx = indexes[j + 1] if j < len(indexes) - 1 else len(
                    eval_pack['pred'][i])
                else:
                    start_idx = indexes[j - 1] + 2 if j > 0 else 0
                    end_idx = indexes[j]
                pred_seq = [int(eval_pack['pred'][i][k]) for k in range(start_idx, end_idx)]
                pred_text = trainer.decode_tokens(pred_seq, self.tokenizer, vocabulary_size)

                # get start and end
                num_bins = 100 # from config
                max_offset = num_bins - 1
                pred_time = [
                    (int(eval_pack['pred'][i][indexes[j]])
                    - vocabulary_size) *
                    eval_pack['duration'][i] / max_offset,
                    (int(eval_pack['pred'][i][indexes[j] + 1]) -
                    vocabulary_size) *
                    eval_pack['duration'][i] / max_offset
                    ]
                
                # if pred_time[1] <= pred_time[0]:  # remove end < start
                #   continue
                last_processed = indexes[j]

                pred.append(pred_text)
                pred_timestamps.append(pred_time)
                
                # round to 2 decimal places
                format_output = "[{x}s, {y}s] ".format(x=np.around(pred_time[0][0]/1000000, decimals=2), y=np.around(pred_time[1][0]/1000000, decimals=2))
                format_output += pred_text
                format_outputs.append(format_output)
            print(format_outputs)
            print("===============================================")
            return format_outputs

class ScenicCall:
    def __init__(self, main, flags):
        self.main = main
        self.FLAGS = flags
        
    def __call__(self):
        return self.run()

    def run(self):
        # Provide access to --jax_backend_target and --jax_xla_backend flags.
        jax.config.config_with_absl()
        run = (functools.partial(self._run_main, main=self.main))
        return run(self.main)
        
    def _run_main(self, argv, *, main):
        """Runs the `main` method after some initial setup."""
        del argv
        # Hide any GPUs form TensorFlow. Otherwise, TF might reserve memory and make
        # it unavailable to JAX.
        tf.config.experimental.set_visible_devices([], 'GPU')

        # Enable wrapping of all module calls in a named_call for easier profiling:
        nn.enable_named_call()

        logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
        logging.info('JAX devices: %r', jax.devices())

        # Add a note so that we can tell which task is which JAX host.
        # (task 0 is not guaranteed to be the host 0)
        platform.work_unit().set_task_status(
            f'host_id: {jax.process_index()}, host_count: {jax.process_count()}')
        if jax.process_index() == 0:
            platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                                self.FLAGS.workdir, 'Workdir')
        self.FLAGS.config.dataset_configs.base_dir = self.FLAGS.data_dir
        rng = jax.random.PRNGKey(self.FLAGS.config.rng_seed)
        logging.info('RNG: %s', rng)

        writer = metric_writers.create_default_writer(
            self.FLAGS.workdir, just_logging=jax.process_index() > 0, asynchronous=True)

        return main(rng=rng, config=self.FLAGS.config, workdir=self.FLAGS.workdir, writer=writer)
