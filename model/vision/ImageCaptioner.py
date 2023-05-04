from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from utils.util import resize_long_edge


class ImageCaptioner:

    def __init__(self, device, base_model='blip2'):
        self.device = device
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        self.base_model = base_model
        self.processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=self.data_type)
        self.model.to(self.device)

    def caption_image(self, image=None, image_src=None):
        if image is None:
            image = Image.open(image_src)
        image = resize_long_edge(image, 384)
        inputs = self.processor(
            images=image, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()
        ret_text = "You see " + generated_text + ".\n"
        # print('\033[1;35m' + '*' * 100 + '\033[0m')
        # print('\nStep1, BLIP2 caption:')
        # print(ret_text)
        # print('\033[1;35m' + '*' * 100 + '\033[0m')
        return ret_text
