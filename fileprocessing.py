import os
import cv2
import json
import numpy as np


def getData230719():
    src_root = '/data/LongAoTian/mnt/192-168-100-77-blood/RBCOverlap4'
    dst_root = '/data/LongAoTian/frameworks/MyULite/data'
    src_images_dir = '/'.join((src_root, 'rbc1024'))
    src_labels_dir = '/'.join((src_root, 'rbc_lbm_labels1024'))
    dst_images_dir = '/'.join((dst_root, 'images'))
    dst_labels_dir = '/'.join((dst_root, 'labels'))
    for p in (dst_images_dir, dst_labels_dir):
        if not os.path.exists(p):
            os.mkdir(p)
                
    classes = ("hollow", "solid")    
    file_names = os.listdir(src_labels_dir)
    for fn in file_names:
        if not fn.endswith('json'):
            continue
        pure_name = fn[:-4]
        image_name = pure_name + 'jpg'
        src_label_path = '/'.join((src_labels_dir, fn))
        src_image_path = '/'.join((src_images_dir, image_name))
        with open(src_label_path, 'r') as f:
            jfile = json.load(f)
        
        lines = []
        selected = False
        if "info" in jfile.keys():  # ISAT
            for obj in jfile["objects"]:
                if obj["category"] == "Selected":
                    selected = True
                    break
            if selected:
                for obj in jfile["objects"]:
                    if obj["category"] not in classes:
                        continue
                    lb = classes.index(obj["category"])
                    coords = [lb] 
                    for pt in obj["segmentation"]:
                        coords.extend(pt)
                    lines.append(" ".join([str(v) for v in coords]))
        else:
            for obj in jfile["shapes"]:
                if obj["label"] == "Selected":
                    selected = True
                    break
            if selected:
                for obj in jfile["shapes"]:
                    if obj["label"] not in classes:
                        continue
                    lb = classes.index(obj["label"])
                    coords = [lb] 
                    for pt in obj["points"]:
                        coords.extend(pt)
                    lines.append(" ".join([str(v) for v in coords]))
        
        if selected:
            dst_image_path = '/'.join((dst_images_dir, image_name))
            dst_label_path = '/'.join((dst_labels_dir, pure_name + 'txt'))
            with open(dst_label_path, 'w') as f:
                f.write("\n".join(lines))
            if not os.path.exists(dst_image_path):
                os.symlink(src_image_path, dst_image_path)
        
       
def getData230725():
    src_root = '/data/LongAoTian/mnt/192-168-100-77-blood/RBCOverlap4'
    dst_root = '/data/LongAoTian/frameworks/MyULite/data/crop_rbc'
    src_images_dir = '/'.join((src_root, 'fimages'))
    src_labels_dir = '/'.join((src_root, 'flabels'))
    dst_images_dir = '/'.join((dst_root, 'images'))
    dst_labels_dir = '/'.join((dst_root, 'labels'))
    for p in (dst_images_dir, dst_labels_dir):
        if not os.path.exists(p):
            os.makedirs(p)
    
    psz = 2
    classes = ("hollow", "solid")    
    file_names = os.listdir(src_labels_dir)
    for fn in file_names:
        if not fn.endswith('json'):
            continue
        pure_name = fn[:-5]
        image_name = pure_name + '.jpg'
        src_label_path = '/'.join((src_labels_dir, fn))
        src_image_path = '/'.join((src_images_dir, image_name))
        image = cv2.imread(src_image_path)
        if image is None:
            continue
        
        with open(src_label_path, 'r') as f:
            jfile = json.load(f)
        
        selected = True
        if "info" in jfile.keys():  # ISAT
            for obj in jfile["objects"]:
                if obj["category"] == "Selected":
                    selected = True
                    break
            if selected:
                height, width = jfile["info"]["height"], jfile["info"]["width"]
                for obj in jfile["objects"]:
                    if obj["category"] not in classes:
                        continue
                    cnt = np.array(obj["segmentation"])
                    bbox = [int(v) for v in obj['bbox']]
                    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width - 1 or bbox[3] == height - 1:
                        continue
                    pbox = [max(0, bbox[0]-psz), max(0, bbox[1]-psz), 
                            min(width, bbox[2]+psz), min(height, bbox[3]+psz)]                    
                    cnt[:, 0] = cnt[:, 0] - pbox[0]
                    cnt[:, 1] = cnt[:, 1] - pbox[1]
                    label = [classes.index(obj["category"])] + cnt.flatten().tolist()
                    crop_image = image[pbox[1]:pbox[3], pbox[0]:pbox[2]]
                    
                    dst_name = "-".join([pure_name] + [str(v) for v in pbox])
                    dst_image_path = "/".join((dst_images_dir, dst_name + ".png"))
                    dst_label_path = "/".join((dst_labels_dir, dst_name + ".txt"))
                    cv2.imwrite(dst_image_path, crop_image)                                        
                    with open(dst_label_path, 'w') as f:
                        f.write(" ".join([str(v) for v in label]))
        else:
            for obj in jfile["shapes"]:
                if obj["label"] == "Selected":
                    selected = True
                    break
            if selected:
                height, width = jfile["imageHeight"], jfile["imageWidth"]
                for obj in jfile["shapes"]:
                    if obj["label"] not in classes:
                        continue
                    cnt = np.array(obj["points"])
                    bbox = [cnt[:, 0].min(), cnt[:, 1].min(), 
                            cnt[:, 0].max(), cnt[:, 1].max()]
                    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == width - 1 or bbox[3] == height - 1:
                        continue
                    pbox = [max(0, bbox[0]-psz), max(0, bbox[1]-psz), 
                            min(width, bbox[2]+psz), min(height, bbox[3]+psz)]                    
                    cnt[:, 0] = cnt[:, 0] - pbox[0]
                    cnt[:, 1] = cnt[:, 1] - pbox[1]
                    label = [classes.index(obj["label"])] + cnt.flatten().tolist()
                    crop_image = image[pbox[1]:pbox[3], pbox[0]:pbox[2]]
                    
                    dst_name = "-".join([pure_name] + [str(v) for v in pbox])
                    dst_image_path = "/".join((dst_images_dir, dst_name + ".png"))
                    dst_label_path = "/".join((dst_labels_dir, dst_name + ".txt"))
                    cv2.imwrite(dst_image_path, crop_image)                                        
                    with open(dst_label_path, 'w') as f:
                        f.write(" ".join([str(v) for v in label]))
        
        
def splitTrainValid():
    data_root = '/data/LongAoTian/frameworks/MyULite/data/crop_rbc'
    # images_dir = '/'.join((data_root, 'images'))
    labels_dir = '/'.join((data_root, 'labels'))
    file_names = os.listdir(labels_dir)
    pure_names = []
    for fn in file_names:
        if fn.endswith('txt'):
            pure_names.append(fn[:-4])
    classes = ("hollow", "solid")
    train_path = '/'.join((data_root, 'train.txt'))
    valid_path = '/'.join((data_root, 'valid.txt'))
    test_path = '/'.join((data_root, 'test.txt'))
    class_path = '/'.join((data_root, 'classes.txt'))
    train_num = round(len(pure_names) * 0.9)
    with open(train_path, 'w') as f:
        f.write("\n".join(pure_names[:train_num]))
    with open(valid_path, "w") as f:
        f.write("\n".join(pure_names[train_num:]))    
    with open(test_path, "w") as f:
        f.write("\n".join(pure_names))    
    with open(class_path, 'w') as f:
        f.write("\n".join(classes))
    
    file_names = os.listdir(data_root)
    for fn in file_names:
        if fn.endswith(".cache"):
            path = '/'.join((data_root, fn))
            os.remove(path)
                        

def getCropChula230725():
    src_root = '/data/LongAoTian/frameworks/MyULite/data/chula'
    dst_root = '/data/LongAoTian/frameworks/MyULite/data/crop_chula'
    src_images_dir = src_root
    src_labels_dir = src_root
    dst_images_dir = '/'.join((dst_root, 'images'))
    dst_labels_dir = '/'.join((dst_root, 'labels'))
    for p in (dst_images_dir, dst_labels_dir):
        if not os.path.exists(p):
            os.makedirs(p)
    
    psz = 2
    border = 5
    classes = ("hollow", "solid")    
    file_names = os.listdir(src_labels_dir)
    for fn in file_names:
        if not fn.endswith('json'):
            continue
        pure_name = fn[:-5]
        image_name = pure_name + '.jpg'
        src_label_path = '/'.join((src_labels_dir, fn))
        src_image_path = '/'.join((src_images_dir, image_name))
        image = cv2.imread(src_image_path)
        if image is None:
            continue
        
        with open(src_label_path, 'r') as f:
            jfile = json.load(f)
        
        selected = False
        if "info" in jfile.keys():  # ISAT
            for obj in jfile["objects"]:
                if obj["category"] == "Selected":
                    selected = True
                    break
            if selected:
                height, width = jfile["info"]["height"], jfile["info"]["width"]
                for obj in jfile["objects"]:
                    if obj["category"] not in classes:
                        continue
                    cnt = np.array(obj["segmentation"])
                    bbox = [int(v) for v in obj['bbox']]
                    if bbox[0] <= border or bbox[1] <= broder or bbox[2] >= width - border or bbox[3] >= height - border:
                        continue
                    pbox = [max(0, bbox[0]-psz), max(0, bbox[1]-psz), min(width, bbox[2]+psz), min(height, bbox[3]+psz)]                    
                    cnt[:, 0] = cnt[:, 0] - pbox[0]
                    cnt[:, 1] = cnt[:, 1] - pbox[1]
                    label = [classes.index(obj["category"])] + cnt.flatten().tolist()
                    crop_image = image[pbox[1]:pbox[3], pbox[0]:pbox[2]]
                    
                    dst_name = "-".join([pure_name] + [str(v) for v in pbox])
                    dst_image_path = "/".join((dst_images_dir, dst_name + ".png"))
                    dst_label_path = "/".join((dst_labels_dir, dst_name + ".txt"))
                    cv2.imwrite(dst_image_path, crop_image)                                        
                    with open(dst_label_path, 'w') as f:
                        f.write(" ".join([str(v) for v in label]))
        else:
            for obj in jfile["shapes"]:
                if obj["label"] == "Selected":
                    selected = True
                    break
            if selected:
                height, width = jfile["imageHeight"], jfile["imageWidth"]
                for obj in jfile["shapes"]:
                    if obj["label"] not in classes:
                        continue
                    cnt = np.array(obj["points"])
                    bbox = [cnt[:, 0].min(), cnt[:, 1].min(), 
                            cnt[:, 0].max(), cnt[:, 1].max()]
                    if bbox[0] <= border or bbox[1] <= border or bbox[2] >= width - border or bbox[3] >= height - border:
                        continue
                    pbox = [max(0, bbox[0]-psz), max(0, bbox[1]-psz), min(width, bbox[2]+psz), min(height, bbox[3]+psz)]                    
                    cnt[:, 0] = cnt[:, 0] - pbox[0]
                    cnt[:, 1] = cnt[:, 1] - pbox[1]
                    label = [classes.index(obj["label"])] + cnt.flatten().tolist()
                    crop_image = image[pbox[1]:pbox[3], pbox[0]:pbox[2]]
                    
                    dst_name = "-".join([pure_name] + [str(v) for v in pbox])
                    dst_image_path = "/".join((dst_images_dir, dst_name + ".png"))
                    dst_label_path = "/".join((dst_labels_dir, dst_name + ".txt"))
                    cv2.imwrite(dst_image_path, crop_image)                                        
                    with open(dst_label_path, 'w') as f:
                        f.write(" ".join([str(v) for v in label]))   


def getChula230728():
    src_root = '/data/LongAoTian/frameworks/MyULite/data/chula'
    src_dir = "/".join((src_root, "Dataset"))
    dst_image_dir = "/".join((src_root, "images"))
    dst_label_dir = "/".join((src_root, "labels"))
    classes = ("hollow", "solid")    
    
    for p in (dst_image_dir, dst_label_dir):
        if not os.path.exists(p):
            os.mkdir(p)
           
    file_names = os.listdir(src_dir)
    for fn in file_names:
        if not fn.endswith('json'):
            continue
        pure_name = fn[:-4]
        image_name = pure_name + 'jpg'
        src_label_path = '/'.join((src_dir, fn))
        src_image_path = '/'.join((src_dir, image_name))
        with open(src_label_path, 'r') as f:
            jfile = json.load(f)
        
        lines = []
        selected = False
        if "info" in jfile.keys():  # ISAT
            for obj in jfile["objects"]:
                if obj["category"] == "Selected":
                    selected = True
                    break
            if selected:
                for obj in jfile["objects"]:
                    if obj["category"] not in classes:
                        continue
                    lb = classes.index(obj["category"])
                    coords = [lb] 
                    for pt in obj["segmentation"]:
                        coords.extend(pt)
                    lines.append(" ".join([str(v) for v in coords]))
        else:
            for obj in jfile["shapes"]:
                if obj["label"] == "Selected":
                    selected = True
                    break
            if selected:
                for obj in jfile["shapes"]:
                    if obj["label"] not in classes:
                        continue
                    lb = classes.index(obj["label"])
                    coords = [lb] 
                    for pt in obj["points"]:
                        coords.extend(pt)
                    lines.append(" ".join([str(v) for v in coords]))
        
        if selected:
            dst_label_path = '/'.join((dst_label_dir, pure_name + 'txt'))
            dst_image_path = '/'.join((dst_image_dir, image_name))
            with open(dst_label_path, 'w') as f:
                f.write("\n".join(lines))      
            if not os.path.exists(dst_image_path):
                os.symlink(src_image_path, dst_image_path)
                

if __name__ == '__main__':
    # getData230719()
    getData230725()
    # getCropChula230725()
    # getChula230728()
    splitTrainValid()
