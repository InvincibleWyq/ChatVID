import os
from model.vision.grit_src.image_dense_captions import image_caption_api


class DenseCaptioner():

    def __init__(self, device):
        self.device = device

    def initialize_model(self):
        pass

    def image_dense_caption(self, image_src, image=None):
        if image is not None:
            dense_caption = image_caption_api(None, self.device, image)
        else:
            dense_caption = image_caption_api(image_src, self.device)
        ret_text = "You find " + dense_caption + "\n"
        # print('\033[1;35m' + '*' * 100 + '\033[0m')
        # print("Step2, Dense Caption:\n")
        # print(ret_text)
        # print('\033[1;35m' + '*' * 100 + '\033[0m')
        return ret_text