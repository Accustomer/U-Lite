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

# Image size for loading and caching
# LOAD_IMAGE_SIZE = (480, 640)  # (height, width)
# LOAD_IMAGE_SIZE = (64, 64)  # (height, width)
LOAD_IMAGE_SIZE = (-1, -1)
# Image formats supported
IMAGE_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')
# Methods for interpolation
CV_INTER_TYPES = (cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_BITS, cv2.INTER_BITS2, 
                  cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LINEAR_EXACT, cv2.INTER_LANCZOS4)    
PIL_INTER_TYPES = (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, 
                   Image.Resampling.BOX, Image.Resampling.HAMMING, Image.Resampling.LANCZOS)


def load_image(self, index):
    image = self.images[index]
    labels = self.labels[index]
    contours = self.contours[index]
    centers = self.centers[index]
    if image is None or labels is None or contours is None or centers is None:        
        name = self.files[index]
        for ifm in IMAGE_FORMATS:
            image_path = '/'.join((self.image_dir, f"{name}.{ifm}"))
            if os.path.exists(image_path):
                break
        label_path = '/'.join((self.label_dir, name + '.txt'))
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        if tuple(image.shape[:2]) != LOAD_IMAGE_SIZE:
            image = cv2.resize(image, (LOAD_IMAGE_SIZE[1], LOAD_IMAGE_SIZE[0]))
        with open(label_path, 'r') as f:
            lines = f.read().strip().splitlines()
        labels, contours, centers = [], [], []
        for line in lines:
            parts = line.split(' ')
            labels.append(int(parts[0]))
            cnt = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape((-1, 2))
            cnt[:, 0] = cnt[:, 0] / w * LOAD_IMAGE_SIZE[1]
            cnt[:, 1] = cnt[:, 1] / h * LOAD_IMAGE_SIZE[0]
            mm = cv2.moments(cnt)
            x = mm['m10'] / mm['m00']
            y = mm['m01'] / mm['m00']
            contours.append(cnt)
            centers.append([x, y])
        centers = np.array(centers, dtype=np.float32)    
        
        self.images[index] = image
        self.labels[index] = labels
        self.contours[index] = contours
        self.centers[index] = centers
    
    h, w = image.shape[:2]
    
    if self.augment and random.random() < self.hyp['gray']:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.merge((gray, gray, gray))
        return gray, labels.copy(), contours.copy(), centers.copy(), h, w
    return image.copy(), labels.copy(), contours.copy(), centers.copy(), h, w


class YoloData(Dataset):
    """ 
    YOLO style Dataset. 
    data/
        images/
            xx.jpg
            yy.jpg
        labels/
            xx.txt   (c1 x1 y1 x2 y2... \nc2 x1` y1` x2` y2`...)
            yy.txt
        classes.txt (c1\nc2)
        train.txt   (xx\nyy\n...)
        valid.txt   (xx\nyy\n...)
        test.txt   (xx\nyy\n...)
    """
    
    def __init__(self, file, img_size, hyp=None, augment=False, cache=False) -> None:
        super().__init__()
        self.hyp = hyp
        self.augment = augment
        self.files = []
        self.img_size = (img_size, img_size) if type(img_size) is int else img_size # (height, width)
        
        global LOAD_IMAGE_SIZE
        if LOAD_IMAGE_SIZE[0] == -1 or LOAD_IMAGE_SIZE[1] == -1:
            LOAD_IMAGE_SIZE = self.img_size    
        
        print('Verify images and labels...')
        data_root = file[:file.rfind('/')]
        self.image_dir = '/'.join((data_root, 'images'))
        self.label_dir = '/'.join((data_root, 'labels'))
        cache_file = file.replace('txt', 'cache')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                base_names = f.read().strip().splitlines()
            n = len(base_names)
            nbar = tqdm(enumerate(base_names))  # progress bar
            for i, name in nbar:                
                self.files.append(name)
                nbar.set_description(f'{i}/{n} sample verified.')
        else:
            with open(file, 'r') as f:
                base_names = f.read().strip().splitlines()
            n = len(base_names)
            nbar = tqdm(enumerate(base_names))  # progress bar
            for i, name in nbar:
                support = False
                for ifm in IMAGE_FORMATS:
                    image_path = '/'.join((self.image_dir, f"{name}.{ifm}"))
                    if os.path.exists(image_path):
                        support = True
                        break
                if not support:
                    print('Image:', name, 'unsupported format!')
                label_path = '/'.join((self.label_dir, name + '.txt'))
                if not os.path.exists(label_path):
                    print('Label:', name, 'does not exist!')
                    continue
                image = cv2.imread(image_path)
                if image is None:
                    print('Image:', name, 'read error!')
                    continue
                self.files.append(name)
                nbar.set_description(f'{i}/{n} sample verified.')
            with open(cache_file, 'w') as f:
                f.write('\n'.join(self.files))

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        n = len(self.files)
        self.images = [None] * n
        self.labels = [None] * n
        self.contours = [None] * n
        self.centers = [None] * n
        self.indices = range(n)
        if cache:
            gb = 0  # Gigabytes of cached imagesx
            results = ThreadPool(8).imap(lambda x: load_image(self, x), self.indices)  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, _ in pbar:
                gb += self.images[i].nbytes
                pbar.desc = f'Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
            
        hyp = self.hyp
        mosaic = self.augment and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels, contours, centers = load_mosaic(self, index)
            else:
                img, labels, contours, centers = load_mosaic9(self, index)
        else:
            img, labels, contours, centers, _, _ = load_image(self, index)
            
        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels, contours, centers = random_perspective(img, labels, contours, centers, 
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
                
            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
              
        # resize
        nobjs = len(labels)
        h, w = img.shape[:2]
        nh, nw = self.img_size
        if h != nh or w != nw:
            if np.random.random() > 0.5:
                img = Image.fromarray(img)
                img = img.resize((nw, nh), resample=np.random.choice(PIL_INTER_TYPES))
                img = np.array(img)
            else:
                img = cv2.resize(img, (nw, nh), interpolation=np.random.choice(CV_INTER_TYPES))
        if nobjs:
            centers = norm_coord(centers, h, w)
            contours = [norm_coord(cnt.astype(np.float32), h, w) for cnt in contours]
        
        # Flip
        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nobjs:
                    for x in (centers, *contours):
                        x[:, 1] = 1 - x[:, 1]
                    
            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nobjs:
                    for x in (centers, *contours):
                        x[:, 0] = 1 - x[:, 0]
                    
        # Mask
        masks = np.zeros((hyp['num_classes'], nh, nw), dtype=np.uint8)
        # dists = []
        if nobjs:
            for i, cnt in enumerate(contours):
                cnt[:, 0] = np.round(cnt[:, 0] * self.img_size[1])
                cnt[:, 1] = np.round(cnt[:, 1] * self.img_size[0])
                idx = labels[i] if hyp['num_classes'] > 1 else 0
                cv2.drawContours(masks[idx], [cnt.astype(np.int32)], -1, 255, cv2.FILLED)
            # dists = np.array([cv2.distanceTransform(masks[i], cv2.DIST_L2, 5) for i in range(hyp['num_classes'])])
            # dists[dists > 0] += 3.6  # sigmoid(4.6) -> 0.99
            
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bxhxw
        img = np.ascontiguousarray(img)
        
        # # Merge labels and centers
        # lwc_tensor = torch.zeros((nobjs, 4))
        # if nobjs:
        #     lwc_tensor[:, 1] = torch.Tensor(labels)
        #     lwc_tensor[:, 2:] = torch.from_numpy(centers)
                    
        # To Tensor
        # labels_tensor = torch.Tensor(labels).long()
        # centers_tensor = torch.from_numpy(centers).float()
        img_tensor = torch.from_numpy(img).float() / 255.0
        mask_tensor = torch.from_numpy(masks).float() / 255
        # dist_tensor = torch.from_numpy(dists).float()
    
        return img_tensor, mask_tensor  # , dist_tensor  # , lwc_tensor
    
    @staticmethod
    def collate_fn(batch):
        """ Return batch images and targets. """
        # images, masks, lwc = zip(*batch)
        # images, masks, dists = zip(*batch)
        images, masks = zip(*batch)
        images = torch.stack(images, 0)  # Batch, channel, height, width
        masks = torch.stack(masks, 0)   # Batch, height, width
        # dists = torch.stack(dists, 0)   # Batch, height, width
        # for i, l in enumerate(lwc):
        #     l[:, 0] = i  # add target image index for build_targets()
        # lwc = torch.cat(lwc, 0)
        return images, masks    # , dists    # , lwc
            
        
def pad_coord(coord, px, py):
    coord[:, 0] += px
    coord[:, 1] += py
    return coord

       
def load_mosaic(self, index):
    # loads images in a 4-mosaic
    labels4, contours4, centers4, totalin = [], [], [], []
    h0, w0 = LOAD_IMAGE_SIZE
    xc = np.random.randint(w0//2, w0 + w0//2)
    yc = np.random.randint(h0//2, h0 + h0//2)
    indices = [index] + random.choices(self.indices, k=3)
    for i, idx in enumerate(indices):
        # Load image
        img, labels, contours, centers, h, w = load_image(self, idx)
        
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((h0 * 2, w0 * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w0 * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h0 * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w0 * 2), min(h0 * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
        
        pad_centers = pad_coord(centers.copy(), padw, padh)
        pad_contours = [pad_coord(cnt.copy(), padw, padh) for cnt in contours]
        pad_totalin = [(cnt[:, 0] > x1a).all() and (cnt[:, 0] < x2a - 1).all() and  
                       (cnt[:, 1] > y1a).all() and (cnt[:, 1] < y2a - 1).all()
                       for cnt in pad_contours]
        labels4.extend(labels)
        contours4.extend(pad_contours)
        centers4.append(pad_centers)
        totalin.extend(pad_totalin)
    
    # Concat/clip labels
    # labels4 = np.concatenate(labels4, 0)
    centers4 = np.vstack(centers4)    
    center_inflags = np.bitwise_and(np.bitwise_and(centers4[:, 0] >= 0, centers4[:, 0] < 2 * w0), 
                                    np.bitwise_and(centers4[:, 1] >= 0, centers4[:, 1] < 2 * h0))
    centers4 = centers4[center_inflags]
    new_labels4, new_contours4, new_totalin = [], [], []
    for i, x in enumerate(contours4):
        if center_inflags[i]:
            x[:, 0] = np.clip(x[:, 0], 0, 2 * w0, out=x[:, 0])
            x[:, 1] = np.clip(x[:, 1], 0, 2 * h0, out=x[:, 1])
            new_contours4.append(x)
            new_labels4.append(labels4[i])
            new_totalin.append(totalin[i])
    contours4 = new_contours4
    labels4 = new_labels4
    totalin = new_totalin
        
    # Show
    # mark = img4.copy()
    # for i in range(len(centers4)):
    #     pt = (round(centers4[i][0]), round(centers4[i][1]))
    #     cv2.circle(mark, pt, 3, (0, 255, 255), 2)
    #     cnt = contours4[i]
    #     n = len(cnt)
    #     for j in range(n):
    #         pt0 = (round(cnt[j][0]), round(cnt[j][1]))
    #         pt1 = (round(cnt[(j+1)%n][0]), round(cnt[(j+1)%n][1]))
    #         cv2.line(mark, pt0, pt1, (0, 255, 255), 2)            
    # cv2.imwrite("temp4.png", mark)
    
    # Augment
    img4, labels4, contours4, centers4 = copy_paste(img4, labels4, contours4, centers4, totalin, probability=self.hyp['copy_paste'])
    # mark = img4.copy()
    # for i in range(np.sum(center_inflags), len(centers4)):
    #     pt = (round(centers4[i][0]), round(centers4[i][1]))
    #     cv2.circle(mark, pt, 3, (0, 0, 255), 2)
    #     cnt = contours4[i]
    #     n = len(cnt)
    #     for j in range(n):
    #         pt0 = (round(cnt[j][0]), round(cnt[j][1]))
    #         pt1 = (round(cnt[(j+1)%n][0]), round(cnt[(j+1)%n][1]))
    #         cv2.line(mark, pt0, pt1, (0, 0, 255), 2)            
    # cv2.imwrite("temp4.png", mark)
    
    img4, labels4, contours4, centers4 = random_perspective(img4, labels4, contours4, centers4, 
                                    degrees=self.hyp['degrees'],
                                    translate=self.hyp['translate'],
                                    scale=self.hyp['scale'],
                                    shear=self.hyp['shear'],
                                    perspective=self.hyp['perspective'],
                                    border=[-h0//2, -w0//2])  # border to remove

    return img4, labels4, contours4, centers4
    
    
def load_mosaic9(self, index):
    # loads images in a 9-mosaic
    labels9, contours9, centers9 = [], [], []
    totalin = []
    oh, ow = LOAD_IMAGE_SIZE
    indices = [index] + random.choices(self.indices, k=8)
    for i, idx in enumerate(indices):
        # Load image
        img, labels, contours, centers, h, w = load_image(self, idx)
        
        # place img in img9
        if i == 0:  # center
            img9 = np.full((oh * 3, ow * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = ow, oh, ow + w, oh + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = ow, oh - h, ow + w, oh
        elif i == 2:  # top right
            c = ow + wp, oh - h, ow + wp + w, oh
        elif i == 3:  # right
            c = ow + w0, oh, ow + w0 + w, oh + h
        elif i == 4:  # bottom right
            c = ow + w0, oh + hp, ow + w0 + w, oh + hp + h
        elif i == 5:  # bottom
            c = ow + w0 - w, oh + h0, ow + w0, oh + h0 + h
        elif i == 6:  # bottom left
            c = ow + w0 - wp - w, oh + h0, ow + w0 - wp, oh + h0 + h
        elif i == 7:  # left
            c = ow - w, oh + h0 - h, ow, oh + h0
        elif i == 8:  # top left
            c = ow - w, oh + h0 - hp - h, ow, oh + h0 - hp

        padw, padh = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords
        img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]  # img9[ymin:ymax, xmin:xmax]
        
        pad_centers = pad_coord(centers.copy(), padw, padh)
        pad_contours = [pad_coord(cnt.copy(), padw, padh) for cnt in contours]
        pad_totalin = [(cnt[:, 0] > x1).all() and (cnt[:, 0] < x2 - 1).all() and  
                       (cnt[:, 1] > y1).all() and (cnt[:, 1] < y2 - 1).all()
                       for cnt in pad_contours]
        labels9.extend(labels.copy())
        contours9.extend(pad_contours)
        centers9.append(pad_centers)
        totalin.extend(pad_totalin)
        
        hp, wp = h, w  # height, width previous
        
   # Offset
    xc = np.random.randint(0, ow)   # min(ow, max(0, np.random.randint(-ow//2, ow+ow//2)))
    yc = np.random.randint(0, oh)   #min(oh, max(0, np.random.randint(-oh//2, oh+oh//2)))
    img9 = img9[yc:yc + 2 * oh, xc:xc + 2 * ow]
    centers9 = np.vstack(centers9)
    centers9 = pad_coord(centers9, -xc, -yc)
    contours9 = [pad_coord(cnt, -xc, -yc) for cnt in contours9]
    
    # Filter outside
    center_inflags = np.bitwise_and(np.bitwise_and(centers9[:, 0] >= 0, centers9[:, 0] < 2 * ow), 
                                    np.bitwise_and(centers9[:, 1] >= 0, centers9[:, 1] < 2 * oh))
    centers9 = centers9[center_inflags]
    new_totalin, new_labels9, new_contours9 = [], [], []
    for i, x in enumerate(contours9):
        if center_inflags[i]:
            x[:, 0] = np.clip(x[:, 0], 0, 2 * ow, out=x[:, 0])
            x[:, 1] = np.clip(x[:, 1], 0, 2 * oh, out=x[:, 1])
            new_contours9.append(x)
            new_labels9.append(labels9[i])
            new_totalin.append(totalin[i])
    contours9 = new_contours9
    labels9 = new_labels9
    totalin = new_totalin
        
    # Show
    # for i in range(len(centers9)):
    #     pt = (round(centers9[i][0]), round(centers9[i][1]))
    #     cv2.circle(img9, pt, 3, (0, 255, 255), 2)
    #     cnt = contours9[i]
    #     n = len(cnt)
    #     for j in range(n):
    #         pt0 = (round(cnt[j][0]), round(cnt[j][1]))
    #         pt1 = (round(cnt[(j+1)%n][0]), round(cnt[(j+1)%n][1]))
    #         cv2.line(img9, pt0, pt1, (0, 255, 255), 2)            
    # cv2.imwrite("temp9.png", img9)
        
    # Augment
    img9, labels9, contours9, centers9 = copy_paste(img9, labels9, contours9, centers9, totalin, probability=self.hyp['copy_paste'])
    # for i in range(np.sum(center_inflags), len(centers9)):
    #     pt = (round(centers9[i][0]), round(centers9[i][1]))
    #     cv2.circle(img9, pt, 3, (0, 0, 255), 2)
    #     cnt = contours9[i]
    #     n = len(cnt)
    #     for j in range(n):
    #         pt0 = (round(cnt[j][0]), round(cnt[j][1]))
    #         pt1 = (round(cnt[(j+1)%n][0]), round(cnt[(j+1)%n][1]))
    #         cv2.line(img9, pt0, pt1, (0, 0, 255), 2)            
    # cv2.imwrite("temp9.png", img9)
    
    img9, labels9, contours9, centers9 = random_perspective(img9, labels9, contours9, centers9, 
                                    degrees=self.hyp['degrees'],
                                    translate=self.hyp['translate'],
                                    scale=self.hyp['scale'],
                                    shear=self.hyp['shear'],
                                    perspective=self.hyp['perspective'],
                                    border=[-oh//2, -ow//2])  # border to remove

    return img9, labels9, contours9, centers9
    

def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments
    

def copy_paste(img, labels, contours, centers, totalin, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(labels)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, cnt, center = labels[j], contours[j], centers[j]
            if not totalin[j]:
                continue
            flip_center, flip_contour = center.copy(), cnt.copy()
            flip_center[0] = w - flip_center[0]
            flip_contour[:, 0] = w - flip_contour[:, 0]
            overlapped = False
            for k, contour in enumerate(contours):
                cond1 = cv2.pointPolygonTest(contour, flip_center, measureDist=False) > 0
                cond2 = cv2.pointPolygonTest(flip_contour, centers[k], measureDist=False) > 0
                if cond1 or cond2:
                    overlapped = True
                    break
            if not overlapped:
                labels.append(l)
                contours.append(flip_contour)
                centers = np.vstack((centers, flip_center))
                cv2.drawContours(im_new, [np.round(cnt).astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img, labels, contours, centers


def random_perspective(img, labels=(), contours=(), centers=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    if degrees == 0 and translate == 0 and scale == 0 and shear == 0 and perspective == 0:
        return img, labels, contours, centers
    
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(labels)
    if n:
        temp = np.ones((len(centers), 3))
        temp[:, :2] = centers
        temp = temp @ M.T
        centers = temp[:, :2] / temp[:, 2:3] if perspective else temp[:, :2]  # perspective rescale or affine
        
        contours = resample_segments(contours)  # upsample
        new_labels, new_contours, new_centers = [], [], []
        for i, contour in enumerate(contours):
            c = centers[i]
            if c[0] < 0 or c[0] >= width or c[1] < 0 or c[1] >= height:
                continue           
            
            xy = np.ones((len(contour), 3))
            xy[:, :2] = contour
            xy = xy @ M.T  # transform
            xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine
            xy[:, 0] = np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            xy[:, 1] = np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            
            new_labels.append(labels[i])
            new_contours.append(xy)
            new_centers.append(c)
            
        labels = new_labels
        contours = new_contours
        centers = np.array(new_centers)

    return img, labels, contours, centers


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def norm_coord(coord, h, w):
    coord[:, 0] = coord[:, 0] / w
    coord[:, 1] = coord[:, 1] / h
    return coord


def get_yolotrain_loader(hyp, args):
    train_file = os.path.join(args.data, 'train.txt')
    train_dataset = YoloData(train_file, args.img_size, hyp, augment=True, cache=args.cache)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, args.workers])  # number of workers
    train_sampler = sampler.RandomSampler(train_dataset, True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=nw, sampler=train_sampler,
                              pin_memory=False, collate_fn=YoloData.collate_fn)
    return train_dataset, train_loader


def get_yoloval_loader(hyp, args, fname='valid.txt', cache=False):
    val_file = os.path.join(args.data, fname)
    val_dataset = YoloData(val_file, args.img_size, hyp, augment=False, cache=args.cache and cache)
    bs = args.val_batch_size if hasattr(args, 'val_batch_size') else args.batch_size
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=args.workers, pin_memory=False,
                            collate_fn=YoloData.collate_fn)
    return val_dataset, val_loader



