import cv2
import torch
import random
import numpy as np
from utils import YoloData

random.seed(8888)


def multiDemo():
    setting_path = '/data/LongAoTian/frameworks/MyULite/data/chula/train.txt'
    hyp_dict = {'num_classes': 1,
                'gray': 0.3, 
                'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
                'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
                'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
                'degrees': 0.0,  # image rotation (+/- deg)
                'translate': 0.2,  # image translation (+/- fraction)
                'scale': 0.75,  # image scale (+/- gain)
                'shear': 0.0,  # image shear (+/- deg)
                'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
                'flipud': 0.0,  # image flip up-down (probability)
                'fliplr': 0.5,  # image flip left-right (probability)
                'mosaic': 1.0,  # image mosaic (probability)
                'copy_paste': 0.5,  # image copy paste (probability)
                'paste_in': 0.15  # image copy paste (probability), use 0 for faster training
                }
    data = YoloData(file=setting_path, img_size=(480, 640), hyp=hyp_dict, augment=True, cache=True)
    for id, (img_tensor, mask_tensor) in enumerate(data):
        print(id)
        img = (img_tensor.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite(f"/data/LongAoTian/frameworks/MyULite/data/temp/{id}-img.jpg", img)
        for i in range(hyp_dict['num_classes']):
            mask = (mask_tensor[i].numpy() * 255).astype(np.uint8)
            cv2.imwrite(f"/data/LongAoTian/frameworks/MyULite/data/temp/{id}-mask{i}.jpg", mask)
            # dist = dist_tensor[i].numpy()
            # dist = (dist / dist.max() * 255).astype(np.uint8)
            # cv2.imwrite(f"/data/LongAoTian/frameworks/MyULite/data/temp/{id}-dist{i}.jpg", dist)
    

def singleDemo():
    setting_path = '/data/LongAoTian/frameworks/MyULite/data/crop_chula/train.txt'
    hyp_dict = {'num_classes': 1,
                'gray': 0.3, 
                'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
                'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
                'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
                'degrees': 0.0,  # image rotation (+/- deg)
                'translate': 0.0,  # image translation (+/- fraction)
                'scale': 0.0,  # image scale (+/- gain)
                'shear': 0.0,  # image shear (+/- deg)
                'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
                'flipud': 0.5,  # image flip up-down (probability)
                'fliplr': 0.5,  # image flip left-right (probability)
                'mosaic': 0.0,  # image mosaic (probability)
                'copy_paste': 0.0,  # image copy paste (probability)
                'paste_in': 0.0  # image copy paste (probability), use 0 for faster training
                }
    data = YoloData(file=setting_path, img_size=64, hyp=hyp_dict, augment=True, cache=False)
    for id, (img_tensor, mask_tensor) in enumerate(data):
        print(id)
        img = (img_tensor.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        mask = (mask_tensor.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite(f"/data/LongAoTian/frameworks/MyULite/data/temp/{id}-img.jpg", img)
        cv2.imwrite(f"/data/LongAoTian/frameworks/MyULite/data/temp/{id}-mask.jpg", mask)
        if id > 10:
            break
    

if __name__ == '__main__':
    multiDemo()
    # singleDemo()
    