# %%
import logging
import flax
import jax
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



def generate(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
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

    train_state = jax_utils.replicate(train_state)
    logging.info('Number of processes is %s', jax.process_count())
    del rng

    tokenizer = trainer.get_tokenizer(config)
    dsname = 'validation'
    iterator = dataset.valid_iter[dsname]
    total_step = range(1)
    for step in total_step:
        batch = next(iterator)
        import functools
        infer_step_pmapped = jax.pmap(
        functools.partial(
            trainer.infer_step,
            model=model,
            config=config,
            debug=config.debug_eval),
        axis_name='batch',
        )
        
        train_state = train_utils.sync_model_state_across_replicas(train_state)
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
        
        _, preds = infer_step_pmapped(train_state, batch) #model, config)
        # import pdb
        # pdb.set_trace()
        eval_pack['pred'] = preds
        eval_pack = jax.tree_map(
            lambda x: x.reshape((np.prod(x.shape[:2]),) + x.shape[2:]), eval_pack)
        
        vocabulary_size = config.dataset_configs.vocabulary_size
        # pred_text = trainer.decode_tokens(preds, tokenizer, vocabulary_size)

        # print(preds, pred_text)
        format_outputs = []
        for i, valid in enumerate(eval_pack['batch_mask']):
            print("===============video[", step, "]====================")
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
            order = config.dataset_configs.order

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
                pred_text = trainer.decode_tokens(pred_seq, tokenizer, vocabulary_size)

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
                format_output = "[{x}s, {y}s]".format(x=np.around(pred_time[0][0]/1000000, decimals=2), y=np.around(pred_time[1][0]/1000000, decimals=2))
                format_output += pred_text
                format_outputs.append(format_output)
            print(format_outputs)
            print("===============================================")
            return format_outputs
