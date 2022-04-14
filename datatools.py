import imgaug as ia
import os
import numpy as np
import cv2
from imgaug import augmenters as iaa
from random import shuffle


class Aug3D:
    def __init__(self):
        aug_list = [iaa.Crop(percent=((0, 0.2))),
                    iaa.SomeOf(2, [iaa.GaussianBlur(sigma=(0, 1.0)),
                                   iaa.LinearContrast(),
                                   iaa.Sharpen()]),
                    iaa.OneOf([iaa.Multiply((0.75, 1.25)), iaa.Add((-50, 50)), ])
                    ]
        self.seq_aug = iaa.Sequential(aug_list)

    def augment_xy(self, img, lmarks):
        det_aug = self.seq_aug.to_deterministic()
        imaug = det_aug.augment_image(img)

        lmarksaug = []
        lmarks = lmarks.reshape(-1, 3)
        for lmark in lmarks:
            lmarksaug.append(ia.Keypoint(x=lmark[0], y=lmark[1]))
        lmarksaug = det_aug.augment_keypoints([ia.KeypointsOnImage(lmarksaug, shape=imaug.shape)])[0]
        if imaug.ndim == 2:
            imaug = np.expand_dims(imaug, -1)

        lmarks[:, :2] = lmarksaug.to_xy_array()
        return imaug, lmarks.flatten()


class Aug2D:
    def __init__(self):
        aug_list = [iaa.Crop(percent=((0, 0.2))),
                    iaa.OneOf([iaa.Multiply((0.75, 1.25)), iaa.Add((-50, 50))])
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


class LoaderBase():
    def __init__(self, data_dir, img_size: tuple,
                 set_size: int = None, valid_size: float = 0.1, used_lmarks: list = None,
                 annotation_file: str = "annotations.txt", augmenter=None):
        """
        Base loader of images dataset. Collected images from inner folders of data_dir directory accordingly with annotation_file.
        :param data_dir: location of images.
        :param img_size: output image size (w, h).
        :param set_size: None for all images, or define specific amount.
        :param valid_size: save proportion for validation. Can be defined 1 if all loader is used for validation.
        :param used_lmarks: define ids of landmarks from annotations to use during training.
        :param annotation_file: each line written as "filename x1 y1 x2 y2 ... xn, yn". Coordinates in pixels.
        :param augmenter: object with method augment_xy(img, landmrks) -> img_augmented, landmarks_augmented.
        """
        self.data_dir = data_dir
        self.lmarks = []
        self.fnames = []
        self.img_size = img_size
        self.aug = augmenter
        self.do_augmentation = True if augmenter is not None else False
        self.used_lmarks = used_lmarks

        with open(os.path.join(data_dir, annotation_file), "r") as f:
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
        np.random.shuffle(keys)
        self.train_set = keys[:-int(set_size * valid_size)] if valid_size else keys
        self.valid_set = keys[-int(set_size * valid_size):] if valid_size else []

    def get_item(self, n):
        raise NotImplementedError("Function must be defined")


class Loader2D(LoaderBase):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def get_item(self, n):
        img = cv2.imread(os.path.join(self.data_dir, self.fnames[n]), 0)
        if img is None:
            print("!!! {}".format(os.path.join(self.data_dir, self.fnames[n])))
            return None, None
        y = self.lmarks[n].copy()
        if self.do_augmentation:
            img, y = self.aug.augment_xy(img, y)
        y[0::2] = y[0::2] / img.shape[1]
        y[1::2] = y[1::2] / img.shape[0]
        if self.used_lmarks is not None:
            y = y[self.used_lmarks]
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32')
        x = np.expand_dims(img, -1)
        return x, y


class Loader3D(LoaderBase):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def get_item(self, n):
        img = cv2.imread(os.path.join(self.data_dir, self.fnames[n]), 0)
        if img is None:
            print("!!! {}".format(os.path.join(self.data_dir, self.fnames[n])))
            return None, None
        y = self.lmarks[n].copy()
        if self.do_augmentation:
            img, y = self.aug.augment_xy(img, y)
        y[0::3] = y[0::3] / img.shape[1]
        y[1::3] = y[1::3] / img.shape[0]
        y[2::3] = y[2::3] / (img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5

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
                    # if not train:
            #    break


def vis_data2d(data_dir):
    with open(os.path.join(data_dir, "annotations.txt"), "r") as f:
        line = f.readline()
        while len(line):
            fname, *vals = line.split()
            vals = list(map(int, vals))
            lmarks = np.array(vals)

            lmarks = lmarks.reshape(-1, 2)

            img = cv2.imread(os.path.join(data_dir, fname))
            for pt in lmarks:
                img = cv2.circle(img, tuple(pt), 1, (0, 0, 0), 3)
            cv2.imshow("face", img)
            k = cv2.waitKey(0)
            if k == 27:
                break
            line = f.readline()


def create_3D_annotations(data_dir):
    import face_alignment

    annfile = open(os.path.join(data_dir, "annotations3D.txt"), "w")
    annfile.close()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cpu")
    fnames = []
    lmarks2D = []

    def get_item(n):
        img = cv2.imread(os.path.join(data_dir, fnames[n]), 0)
        if img is None:
            print("!!! {}".format(os.path.join(data_dir, fnames[n])))
            return None, None
        y = lmarks2D[n].copy()
        return img, y

    with open(os.path.join(data_dir, "annotations.txt"), "r") as f:
        line = f.readline()
        while len(line):
            fname, *vals = line.split()
            vals = list(map(int, vals))
            lmarks2D.append(np.array(vals, dtype=np.float32))
            fnames.append(fname)
            line = f.readline()

    for n in range(len(fnames)):
        img, lm2D = get_item(n)
        lm2D = lm2D.reshape(-1, 2)
        xmin, ymin = lm2D.min(0)
        xmax, ymax = lm2D.max(0)
        lm3D = fa.get_landmarks_from_image(img, detected_faces=[[xmin, ymin, xmax, ymax]])[0]

        print("{:40}".format(fnames[n]), img.shape)
        with open(os.path.join(data_dir, "annotations3D.txt"), "a") as annfile:
            annfile.write(fnames[n])
            for pt in lm3D:
                annfile.write(" {} {} {}".format(*[round(c) for c in pt]))
            annfile.write("\n")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    """
    subpursers = parser.add_subparsers(title="api", help="available actions")
    vis2d = subpursers.add_parser("vis2d")
    vis2d.add_argument("data_dir", type=str, help="path to data with annotations.txt and images")
    create3d = subpursers.add_parser("create3d")#(title="Create 3D landmarks based on 2d annotations")
    create3d.add_argument("data_dir", type=str, help="path to data with annotations.txt and images")    
    """
    parser.add_argument("api", choices=["vis2d", "create3d"], help="available actions")
    parser.add_argument("data_dir", type=str, help="path to data with annotations.txt and images")

    args = parser.parse_args()

    if args.api == "vis2d":
        vis_data2d(args.data_dir)
    elif args.api == "create3d":
        create_3D_annotations(args.data_dir)
