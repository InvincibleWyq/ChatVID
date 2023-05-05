import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor


class ImageCaptioner:

    def __init__(self, device='cuda'):
        self.device = device
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        self.processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=self.data_type).to(self.device)

    def __call__(self, imgs):
        inputs = self.processor(
            images=imgs, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)

        return generated_text
