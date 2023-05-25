import os
from typing import Any, Callable

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.vid2seq import models
from scenic.projects.vid2seq import trainer
from scenic.projects.vid2seq.datasets.dense_video_captioning_tfrecord_dataset import get_datasets
# replace with the path to your JAVA bin location
JRE_BIN_JAVA = "/nvme/wangyiqin/java/jre1.8.0_371/bin/java"

flags.DEFINE_string('jre_path', '',
                    'Path to JRE.')

FLAGS = flags.FLAGS


def get_model_cls(model_name: str) -> Callable[..., Any]:
  """Returns model class given its name."""
  if model_name == 'vid2seq':
    return models.DenseVideoCaptioningModel
  raise ValueError(f'Unrecognized model: {model_name}.')

def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):

    model = models.DenseVideoCaptioningModule(config)
    
    # check model info
    print(model)
    
    # iterate the layers of the model in flax 
    # rng, init_rng = jax.random.split(rng)
    # (params, model_state, num_params,
    # gflops) = train_utils.initialize_model_with_pytree(
    #     model_def=model.flax_model,
    #     input_spec=(encoder_input_spec, decoder_input_spec),
    #     config=config,
    #     rngs=init_rng)
    # logging.info('The model has %d params, uses %d gflops', num_params, gflops or
    #             -1)
    # model.setup()
    # params = model.init(rng)['params']
    # jax.tree_util.tree_map(jnp.shape, params)
    

    
if __name__ == '__main__':
  app.run(main=main)



