import argparse
from model import Captioner, VicunaHandler
import json
from config.config_utils import get_config

if __name__ == '__main__':
    _argparser = argparse.ArgumentParser()
    _argparser.add_argument("--config_path", required=True, type=str)
    _args = _argparser.parse_args()
    config = get_config(_args.config_path)

    captioner = Captioner(config)
    prompted_captions, speech = captioner.caption_frames(
        config['video_path'], config['video_name'], config['fps'])

    with open('./test.json', 'w') as f:
        json.dump(prompted_captions, f)

    handler = VicunaHandler(config['vicuna'])
    handler.summarise_caption(prompted_captions)

    handler.chat()
