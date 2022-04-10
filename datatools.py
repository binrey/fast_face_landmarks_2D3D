import imgaug as ia
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa
from random import shuffle


class Aug():
    def __init__(self, output_size):
        aug_list = [iaa.Crop(percent=((0, 0.2))),
                    #iaa.HorizontalFlip(0.5),
                    #iaa.SomeOf(2, [iaa.GaussianBlur(sigma=(0, 1.0)),
                    #               iaa.LinearContrast(),
                    #               iaa.Sharpen()]),
                    iaa.OneOf([iaa.Multiply((0.75, 1.25)), iaa.Add((-50, 50)),])
                    ]
        self.seq_aug = iaa.Sequential(aug_list)

    def augment_xy(self, img, lmarks):
        det_aug = self.seq_aug.to_deterministic()
        imaug = det_aug.augment_image(img)

        lmarksaug = []
        for lmark in lmarks.reshape(-1, 2):
            lmarksaug.append(ia.Keypoint(x=lmark[0], y=lmark[1]))
        lmarksaug = det_aug.augment_keypoints([ia.KeypointsOnImage(lmarksaug, shape=imaug.shape)])[0]
        if imaug.ndim == 2:
            imaug = np.expand_dims(imaug, -1)

        lmarksaug = lmarksaug.to_xy_array().flatten()
        return imaug, lmarksaug

class Loader300W():
    """
    Loader of 300W dataset. Loader used in generator
    """
    def __init__(self, data_dir, img_size, set_size=None, valid_size=0.1, used_lmarks=None): 
        self.data_dir = data_dir
        self.lmarks = []
        self.fnames = []
        self.img_size = img_size
        self.aug = Aug(img_size)
        self.do_augmentation = True
        self.used_lmarks = used_lmarks
        
        with open(os.path.join(data_dir, "annotations.txt"), "r") as f:
            line = f.readline()
            while len(line):
                fname, *vals = line.split()
                vals = list(map(int, vals))
                lmarks = np.array(vals, dtype=np.float32)
                self.lmarks.append(lmarks)
                self.fnames.append(fname)
                line = f.readline()

        self.lmarks = np.array(self.lmarks)
        set_size = len(self.fnames) if set_size is None else set_size
        keys = np.arange(set_size, dtype=np.int32)
        np.random.seed(0)
        np.random.shuffle(keys)
        self.train_set = keys[:-int(set_size*valid_size)] if valid_size else keys
        self.valid_set = keys[-int(set_size*valid_size):] if valid_size else []
        
    def get_item(self, n):
        img = cv2.imread(os.path.join(self.data_dir, self.fnames[n]), 0)
        if img is None:
            print("!!! {}".format(os.path.join(self.data_dir, self.fnames[n])))
            return None, None
        y = self.lmarks[n].copy()
        if self.do_augmentation:
            img, y = self.aug.augment_xy(img, y)
        y[0::2] = y[0::2]/img.shape[1]
        y[1::2] = y[1::2]/img.shape[0]
        if self.used_lmarks is not None:
            y = y[self.used_lmarks]
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32')
        x = np.expand_dims(img, -1)
        return x, y

    
class Generator(object):
    """
    Generates batches of train or validation images. 
    """
    def __init__(self, loader, batch_size):
        self.batch_size = batch_size
        self.L = loader
        self.lmarks = self.L.lmarks

    def get_iterator(self, train=True):
        while True:
            if train:
                shuffle(self.L.train_set)
                keys = self.L.train_set
                self.L.do_augmentation = True
            else:
                keys = self.L.valid_set
                self.L.do_augmentation = False
            inputs = []
            targets = []

            for key in keys:
                img, y = self.L.get_item(key)
                if y is None:
                    continue
                
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets 
            #if not train:
            #    break
                    
def show_xy(x, y_true=None, y_pred=None):
    plt.imshow(x+0.5, cmap="gray")
    if y_true is not None:
        cmap = np.arange(len(y_true)/2)
        plt.scatter(y_true[0::2]*x.shape[1], y_true[1::2]*x.shape[0], c=cmap, s=100, marker="*")
    if y_pred is not None:  
        cmap = np.arange(len(y_pred)/2)
        plt.scatter(y_pred[0::2]*x.shape[1], y_pred[1::2]*x.shape[0], c=cmap)