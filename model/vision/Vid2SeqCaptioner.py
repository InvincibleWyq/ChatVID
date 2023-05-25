from typing import Any
from model.utils.extract_clip_feature import extract_clip_feature_single_video_fps
from model.utils.generate_tf_record import generate
import torch

class Vid2SeqCaptioner:
    """Vid2SeqCaptioner is a class that uses a video to generate a caption for the video.
    
    Description:
        It uses the Google Scenic Vid2Seq as the base model. Note that Scenic is a project designed to use with TPUs. And GPU resources(70G VMeomry at least) maybe be not enough to run the model. So, we need to use the CPU to run the model.
    """
    def __init__(self, config):
        self.config = config
    
    def __call__(self, video_path):
        self._preprocess(video_path)
        
        
    def _preprocess(self, video_path):
        """Preprocess the video.
        
        Description:
            Pipeline: CLIP -> *.npy -> *.csv(video) -> generate_from_file.py -> file0000-0003.tfrecord
        Args:
            video_path: The path of the video.
        """
        
        # Extract CLIP features first
        device = "cuda" if torch.cuda.is_available() else "cpu"
        video_feat, video_info = extract_clip_feature_single_video_fps(
            video_path=video_path,
            clip_ckpt_path=self.config['clip_path'],
            device=device
        )
        
        video_feat = video_feat.cpu()
        video_feat = video_feat.numpy()
        
        video_info_dict = {
            'basename' : self.config['basename'],
            'output_path' : self.config['output_path'],
            'asr_start' : None,
            'asr_end' : None,
            'asr_string' : None,
            'video_id' : video_path.split('/')[-1].split('.')[0],
            'features' : video_feat,
            'duration' : video_info['total_frames'] / video_info['avg_fps'] * 1000000,
        }
        generate(video_info_dict)
        print("tfrecord file generated at {}".format(self.config['output_path']))
        
        