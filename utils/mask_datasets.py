import os
import cv2
import math
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, sampler
from multiprocessing.pool import ThreadPool


IMAGE_SIZE = (1024, 1024)
CV_INTER_TYPES = (cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_BITS, cv2.INTER_BITS2, 
                  cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LINEAR_EXACT, cv2.INTER_LANCZOS4)
PIL_INTER_TYPES = (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, 
                   Image.Resampling.BOX, Image.Resampling.HAMMING, Image.Resampling.LANCZOS)



class MaskData(Dataset):
    raise NotImplementedError
