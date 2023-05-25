"""Tests for segmentation_datasets."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import ml_collections
from scenic.projects.robust_segvit.datasets import segmentation_variants

EXPECTED_DATASETS = [
    ('ade20k_ind_c', 'ade20k_ind_c', 'gaussian_noise', 1, 'validation'),
]


class SegmentationVariantsTest(parameterized.TestCase):

  @parameterized.named_parameters(EXPECTED_DATASETS)
  def test_available(self, name, corruption_type, corruption_level, val_split):
    """Test we can load a corrupted dataset."""
    num_shards = jax.local_device_count()
    config = ml_collections.ConfigDict()
    config.batch_size = num_shards*2
    config.eval_batch_size = num_shards*2
    config.num_shards = num_shards

    config.rng = jax.random.PRNGKey(0)
    config.dataset_configs = ml_collections.ConfigDict()
    config.dataset_configs.train_target_size = (120, 120)
    if corruption_type:
      config.dataset_configs.name = f'{name}_{corruption_type}_{corruption_level}'
    else:
      config.dataset_configs.name = name
    config.dataset_configs.denoise = None
    config.dataset_configs.use_timestep = 0
    config.dataset_configs.val_split = val_split
    _, dataset, _, _ = segmentation_variants.get_dataset(**config)
    batch = next(dataset)
    self.assertEqual(
        batch['inputs'].shape,
        (num_shards, config.eval_batch_size // num_shards, 120, 120, 3))


if __name__ == '__main__':
  absltest.main()
