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
