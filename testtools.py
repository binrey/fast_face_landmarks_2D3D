import numpy as np
import cv2
import os
from shutil import rmtree


def keras_wrapper(img_crop, model):
    """
    input img_crop: face crop before resize
    output: landmarks as flatten np.array
    """
    net_input = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    net_input = cv2.resize(net_input, model.input_shape[1:-1][::-1])
    net_input = net_input.reshape(1, net_input.shape[0], net_input.shape[1], 1)
    lmarks = model.predict(net_input)
    return lmarks


class CropTester():
    def __init__(self, predict_func, pad_min_max=(0, 0.2), num_iterations=100, crop_type="rand", rseed=None):
        self.predict_func = predict_func
        self.pad_min_max = pad_min_max
        self.num_iterations = num_iterations
        self.lmarks_all = []
        self.rseed = rseed
        assert crop_type in ("rand", "scale")
        self.crop_type = crop_type
        
        np.random.seed(self.rseed)
        
    def test_image(self, img_path, color=(255, 255, 255)):
        rmtree("croptest_out", ignore_errors=True)
        os.makedirs("croptest_out", exist_ok=True)
        self.lmarks_all = []
        np.random.seed(None)
        x = cv2.imread(img_path)
        if x is None:
            print(f"!!! {img_path}")
            return
        for aug_iter in range(self.num_iterations):
            if self.rseed is not None:
                np.random.seed(self.rseed*aug_iter)
            if self.crop_type == "rand":
                pads = np.random.uniform(self.pad_min_max[0], self.pad_min_max[1], 4)
            else:
                pad = aug_iter*(self.pad_min_max[1] - self.pad_min_max[0])/self.num_iterations + self.pad_min_max[0]
                pads = np.array([pad, 0, pad, 0])
            pads[[0, 2]] = (pads[[0, 2]]*x.shape[1]).round(0)
            pads[[1, 3]] = (pads[[1, 3]]*x.shape[0]).round(0)
            pads = pads.astype(np.int32)

            net_input = x[pads[1]:-max(1, pads[3]), pads[0]:-max(1, pads[2])]
            lmarks = self.predict_func(net_input)

            xplot = x.copy()
            box_pt1 = tuple([pads[0], pads[1]])
            box_pt2 = tuple([x.shape[1]-pads[2], x.shape[0]-pads[3]])
            cv2.rectangle(xplot, box_pt1, box_pt2, (0, 255, 255), 1)  

            self.lmarks_all.append([])
            for pt in lmarks.reshape(-1, 2):
                pt_real = [box_pt1[0] + int(pt[0] * (box_pt2[0] - box_pt1[0])),
                           box_pt1[1] + int(pt[1] * (box_pt2[1] - box_pt1[1]))]
                self.lmarks_all[-1] += pt_real
                xplot = cv2.circle(xplot, tuple(pt_real), 2, color, 2)
            cv2.imwrite("croptest_out/{:03d}.png".format(aug_iter), xplot)
        self.lmarks_all = np.array(self.lmarks_all)
        print(">>> Output images are placed in ./croptest_out folder")
        
    def mean_pixel_std(self):
        return self.lmarks_all.std(0).mean()


if __name__ == "__main__":
    from argparse import ArgumentParser
    from tensorflow.keras.models import load_model
    from functools import partial

    parser = ArgumentParser()
    parser.add_argument("--h5", type=str, help="path to h5 model file")
    parser.add_argument("--img", type=str, help="path to image file")
    args = parser.parse_args()

    model = load_model(args.h5)
    print(model.summary())

    kw = partial(keras_wrapper, model=model)
    ct = CropTester(kw)
    ct.test_image(args.img)
    print("\n mean deviation of predictions, px:",  ct.mean_pixel_std())