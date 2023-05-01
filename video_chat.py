import argparse
from model import Captioner, VicunaHandler
from utils.util import display_images_and_text
import os
import json
from config.config_utils import get_config


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser()
    _argparser.add_argument("--config_path", required=True, type=str)
    _args = _argparser.parse_args()
    config = get_config(_args.config_path)

    captioner = Captioner(config)
    prompted_captions = captioner.caption_frames(config['video_path'], config['video_name'], config['fps'])
    
    with open('./test.json', 'w') as f:
        json.dump(prompted_captions, f)
    
    prompted_captions = json.load(open('./test.json'))
    handler = VicunaHandler(config['vicuna'])
    handler.summarise_caption(prompted_captions)
    
    handler.chat()
    
    
    
    
    
    
    
    # get all images names in the image_src folder
    # question_dict = {}
    # image_names = os.listdir(args.image_src)
    # # sort the names in alphabetical order from 0.jpg to 999.jpg
    # image_names.sort()
    
    # frames_target = args.target_frame
    # num_images = len(image_names)
    # # get the number of frames to be processed
    # frame_gap = num_images // frames_target
    
    # with open(args.image_src+'questions.json', 'w') as f:
    #     counter = 0
    #     for name in image_names:
    #         if not name.endswith('.jpg'):
    #             continue
    #         if counter % frame_gap != 0:
    #             counter += 1
    #             continue
    #         counter += 1
    #         question = processor.image_to_question(args.image_src+name)
    #         print("*" * 50)
    #         print(question)
    #         question_dict[name] = question
    #     json.dump(question_dict, f)
    