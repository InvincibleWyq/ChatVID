"""Base training config for CLIP w/ linear + BERT."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_linear_bert as general_base_clip_linear_bert
from scenic.projects.lang4video.configs.train import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = general_base_clip_linear_bert.get_config(run_local)
  config.update(mixin.get_config(run_local))

  config.optimizer_configs.params_to_freeze = (r'^image_encoder/encoders_0/',
                                               r'^text_encoder/')

  return config
