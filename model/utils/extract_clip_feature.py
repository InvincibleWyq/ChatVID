import clip
import numpy as np
import torch
from mmaction.datasets.transforms import (CenterCrop, DecordDecode, DecordInit,
                                          FormatShape, Resize)
from torchvision import transforms


def extract_clip_feature_single_video_fps(
        video_path: str,
        clip_ckpt_path: str = 'ViT-L-14.pt',
        device: str = 'cuda'):

    class SampleFrames1FPS(object):
        '''Sample frames at 1 fps.

        Required Keys:
            - total_frames
            - start_index
            - avg_fps

        Added Keys:
            - frame_interval
            - frame_inds
            - num_clips
        '''

        def transform(self, video_info: dict) -> dict:
            video_info['frame_inds'] = np.arange(
                video_info['start_index'],
                video_info['total_frames'],
                video_info['avg_fps'],
                dtype=int)  # np.arange(start, stop, step, dtype)
            video_info['frame_interval'] = 1
            video_info['num_clips'] = len(video_info['frame_inds'])
            return video_info

    class SampleFrames5FPS(object):
        '''Sample frames at 5 fps.

        Required Keys:
            - total_frames
            - start_index
            - avg_fps

        Added Keys:
            - frame_interval
            - frame_inds
            - num_clips
        '''

        def transform(self, video_info: dict) -> dict:
            video_info['frame_inds'] = np.arange(
                video_info['start_index'],
                video_info['total_frames'],
                video_info['avg_fps'] // 5,
                dtype=int)
            video_info['frame_interval'] = 1
            video_info['num_clips'] = len(video_info['frame_inds'])
            return video_info

    video_info = {'filename': video_path, 'start_index': 0}
    video_processors = [
        DecordInit(),
        SampleFrames1FPS(),
        DecordDecode(),
        Resize(scale=(-1, 224)),
        CenterCrop(crop_size=224),
        FormatShape(input_format='NCHW'),
    ]

    # decode video to imgs
    for processor in video_processors:
        video_info = processor.transform(video_info)

    imgs = torch.from_numpy(video_info['imgs'])  # uint8 img tensor

    imgs_transforms = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
            inplace=False)
    ])

    # uint8 -> float, then normalize
    imgs = imgs_transforms(imgs).to(device)

    # load model
    clip_model, _ = clip.load(clip_ckpt_path, device)

    # encode imgs get features
    with torch.no_grad():
        video_feat = clip_model.encode_image(imgs)

    return video_feat, video_info


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_names = [
        'cook.mp4', 'latex.mp4', 'nba.mp4', 'temple_of_heaven.mp4',
        'south_pole.mp4', 'tv_series.mp4', 'formula_one.mp4', 'make-up.mp4',
        'police.mp4'
    ]
    video_dir = '/mnt/petrelfs/wangyiqin/vid_cap/examples/videos/'

    for video_name in video_names:
        video_feat = extract_clip_feature_single_video_fps(
            video_path=video_dir + video_name,
            clip_ckpt_path='ViT-L-14.pt',
            device=device)
        video_feat = video_feat.cpu()
        # compress to one dimension
        video_feat = video_feat.numpy()

        np.save('clip_features/20/' + video_name[:-4] + '.npy', video_feat)
        print(video_feat.shape)
        print(video_name + ' DONE')
