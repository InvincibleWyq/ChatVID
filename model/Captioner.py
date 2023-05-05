from mmaction.datasets.transforms import (DecordInit, SampleFrames, Resize,
                                          FormatShape, DecordDecode)
from model.audio import SpeechRecognizer
from model.vision import DenseCaptioner, ImageCaptioner


class Captioner:
    """ Captioner class for video captioning
    """

    def __init__(self, config):
        """ Initialize the captioner
        Args:
            config: configuration file
        """
        self.config = config
        self.image_captioner = ImageCaptioner(device=config['device'])
        self.dense_captioner = DenseCaptioner(device=config['device'])
        self.speech_recognizer = SpeechRecognizer(device=config['device'])

        self.src_dir = ''

    def caption_frames(self, video_path, num_frames=8):
        """ Caption all frames in the folder"""
        print("Captioning frames...")

        video_info = {'filename': video_path, 'start_index': 0}

        video_processors = [
            DecordInit(),
            SampleFrames(clip_len=1, frame_interval=1, num_clips=num_frames),
            DecordDecode(),
            Resize(scale=(-1, 720)),
            FormatShape(input_format='NCHW'),
        ]
        for processor in video_processors:
            video_info = processor.transform(video_info)

        timestamp_list = [
            round(i / video_info['avg_fps'], 1)
            for i in video_info['frame_inds']
        ]

        image_captions = self.image_captioner(imgs=video_info['imgs'])
        dense_captions = self.dense_captioner(imgs=video_info['imgs'])
        speech = self.speech_recognizer(video_path)

        overall_captions = ""
        for i in range(num_frames):
            overall_captions += "[" + str(timestamp_list[i]) + " second]: "
            overall_captions += "You see " + image_captions[i]
            overall_captions += "You find " + dense_captions[i] + "\n"

        if speech != "":
            overall_captions += "You hear \"" + speech + "\"\n"

        print("Captions generated")
        return overall_captions
