import os
import cv2
import yaml
import json
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from models import createULite
from utils import intersect_dicts

IMAGE_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')


def inferDemo():
    k = 'crop_chula'
    cfg_path = f'/data/LongAoTian/frameworks/MyULite/runs/{k}/cfg.yaml'
    ckpt_path = f'/data/LongAoTian/frameworks/MyULite/runs/{k}/best.pth.tar'
    
    device = '1'
    if device is not None and device.lower() != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    model = createULite(cfg_dict).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        csd = intersect_dicts(ckpt['state_dict'], model.state_dict())
    else:
        csd = intersect_dicts(ckpt, model.state_dict())
    print("Intersection modules: ", csd.keys())
    model.load_state_dict(csd)
    model.eval()

    # Infer pth
    data_root = f'/data/LongAoTian/frameworks/MyULite/data/{k}'
    src_dir = '/'.join((data_root, 'images'))
    dst_dir = f'/data/LongAoTian/frameworks/MyULite/runs/{k}/prediction'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    # img_size = (640, 512)
    img_size = (64, 64)
    num_limit = 20
    with torch.no_grad():
        file_names = os.listdir((src_dir))
        random.shuffle(file_names)
        fbar = tqdm(enumerate(file_names))
        nfile = len(file_names)
        for i, fn in fbar:
            support = False
            for ifm in IMAGE_FORMATS:
                if fn.endswith(ifm):
                    support = True
                    break
            if not support:
                continue
            pure_name = ".".join(fn.split(".")[:-1])
            src_path = '/'.join((src_dir, fn))
            image = cv2.imread(src_path)
            image = cv2.resize(image, img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_var = torch.from_numpy(image)
            input_var = input_var.permute((2, 0, 1)) / 255.0
            input_var = input_var.unsqueeze_(0).to(device)
            mask_res = model(input_var)
            
            # heatmap
            heatmap0 = mask_res[0][0].cpu().numpy()
            # heatmap1 = mask_res[0][1].cpu().numpy()
            # mask = np.bitwise_or(heatmap0 > 0.5, heatmap1 > 0.5)
            mask = heatmap0 > 0.6
            mask = (mask * 255).astype('uint8')
            mark = np.hstack((image, cv2.merge((mask, mask, mask))))
            save_path = '/'.join((dst_dir, f"{pure_name}-mark.jpg"))
            cv2.imwrite(save_path, mark)
            
            s = f"{i}/{nfile} file: {fn}"
            fbar.set_description(s)
            
            if num_limit > 0 and i >= num_limit:
                break
        fbar.close()


if __name__ == '__main__':
    inferDemo()
