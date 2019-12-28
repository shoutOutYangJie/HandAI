from yaml import load
import yaml
import sys
sys.path.append('../networks/')
from model_mobilenetv2_seg_small import MobileNetV2
import torch
import cv2
import numpy as np
def get_seg_model():
    config_path = '../config/portraitNet.yaml'
    with open(config_path, 'rb') as f:
        cont = f.read()
    cf = load(cont, Loader=yaml.FullLoader)
    portraitNet = MobileNetV2(n_class=2,
                              config=cf)
    bestModelFile = '../ckpts/portraitNet/portraitNet.pth'
    checkpoint_video = torch.load(bestModelFile)
    portraitNet.load_state_dict(checkpoint_video, strict=False)
    portraitNet.cuda()
    return portraitNet

def test_on_image():
    model = get_seg_model()
    img = cv2.imread('../data/example_data/portrait.jpg')
    mask = model.predict(img.copy())

    # blur background
    background = img.copy()
    background = cv2.blur(background, (17, 17))

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blur_img = mask * img + (1 - mask) * background
    blur_img = np.clip(blur_img, 0, 255)
    blur_img = np.uint8(blur_img)
    comparison = np.concatenate((img, blur_img), axis=1)
    cv2.imshow('comparison', comparison)
    cv2.waitKey()

if __name__=='__main__':
    test_on_image()
