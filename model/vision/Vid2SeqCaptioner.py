from typing import Any
from model.utils import extract_clip_feature_single_video_fps, generate, ScenicCall, ScenicModel
from config import vid2seq_config

import torch


import sys, os
from pathlib import Path
# append current path to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent / "scenic"))
from scenic.projects.vid2seq.playground import generate as vid2seq_generate

class Flag(object):
    pass

class Vid2SeqCaptioner:
    """Vid2SeqCaptioner is a class that uses a video to generate a caption for the video.
    
    Description:
        It uses the Google Scenic Vid2Seq as the base model. Note that Scenic is a project designed to use with TPUs. And GPU resources(70G VMeomry at least) maybe be not enough to run the model. So, we need to use the CPU to run the model.
    """
    def __init__(self, config):
        self.config = config
        flags = Flag()
        flags.workdir = self.config['work_dir']
        flags.config = vid2seq_config.get_config()
        # flags.config = self.config['config_path']
        flags.data_dir = self.config['output_path']
        flags.ckpt_dir = self.config['checkpoint_path']
        self.model = ScenicModel(flags)
        
    def __call__(self, video_path):
        self._preprocess(video_path)
        return self.model()
        # call = ScenicCall(vid2seq_generate, flags)
        # return call()
        
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
        
        # get numpy array
        video_feat = video_feat.cpu()
        video_feat = video_feat.numpy()
        
        # create a dict to store the video info
        video_info_dict = {
            'basename' : 'test',
            'output_path' : self.config['output_path'],
            'asr_start' : None,
            'asr_end' : None,
            'asr_string' : None,
            'video_id' : video_path.split('/')[-1].split('.')[0],
            'features' : video_feat,
            'duration' : video_info['total_frames'] / video_info['avg_fps'] * 1000000,
        }
        # begin to generate tfrecord file
        generate(video_info_dict)
        print("tfrecord file generated at {}".format(self.config['output_path']))
        
        