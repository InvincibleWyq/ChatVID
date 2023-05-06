from model.vision.grit_src.image_dense_captions import image_caption_api
import cv2


class DenseCaptioner():

    def __init__(self, device):
        self.device = device

    def __call__(self, imgs):
        dense_captions = []
        for img in imgs:
            cv2_img = cv2.merge([img[2], img[1], img[0]])  # BGR
            caption = image_caption_api(cv2_img, device=self.device)
            dense_captions.append(caption)

        return dense_captions
