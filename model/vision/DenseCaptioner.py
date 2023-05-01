import os
from model.vision.grit_src.image_dense_captions import image_caption_api

class DenseCaptioner():
    def __init__(self, device):
        self.device = device


    def initialize_model(self):
        pass
    
    def image_dense_caption(self, image_src):
        dense_caption = image_caption_api(image_src, self.device)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print("Step2, Dense Caption:\n")
        print(dense_caption)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return dense_caption
    