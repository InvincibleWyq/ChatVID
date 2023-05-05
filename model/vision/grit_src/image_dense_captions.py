import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image, _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'model/vision/grit_src/third_party/CenterNet2/projects/CenterNet2/')
from model.vision.grit_src.third_party.CenterNet2.projects.CenterNet2.centernet.config import add_centernet_config
from model.vision.grit_src.grit.config import add_grit_config

from model.vision.grit_src.grit.predictor import VisualizationDemo
import json
from utils.util import resize_long_edge_cv2


# constants
WINDOW_NAME = "GRiT"

def dense_pred_to_caption_no_bbox(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ""
    for i in range(len(object_description) - 1):
        new_caption += (object_description[i] + ", ")
    new_caption += (object_description[-1] + ".")
    return new_caption

def dense_pred_to_caption(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ""
    for i in range(len(object_description)):
        new_caption += (object_description[i] + ": " + str([int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
    return new_caption

def setup_cfg(args):
    cfg = get_cfg()
    if args["cpu"]:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    if args["test_task"]:
        cfg.MODEL.TEST_TASK = args["test_task"]
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser(device):
    arg_dict = {'config_file': "model/vision/grit_src/configs/GRiT_B_DenseCap_ObjectDet.yaml", 'cpu': False, 'confidence_threshold': 0.5, 'test_task': 'DenseCap', 'opts': ["MODEL.WEIGHTS", "pretrained_models/grit_b_densecap_objectdet.pth"]}
    if device == "cpu":
        arg_dict["cpu"] = True
    return arg_dict

def image_caption_api(image_src, device, image=None):
    args2 = get_parser(device)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image: # from PIL.Image
        img = _apply_exif_orientation(image)
        img = convert_PIL_to_numpy(img, format="BGR")
        img = resize_long_edge_cv2(img, 384)
        predictions, visualized_output = demo.run_on_image(img)
        new_caption = dense_pred_to_caption_no_bbox(predictions)
    else:
        img = read_image(image_src, format="BGR")
        img = resize_long_edge_cv2(img, 384)
        predictions, visualized_output = demo.run_on_image(img)
        new_caption = dense_pred_to_caption(predictions)
    return new_caption