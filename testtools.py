import numpy as np
import cv2
import os
import math
from shutil import rmtree
from camera_reader import create_camera_reader
from matplotlib import pyplot as plt


def keras_wrapper(img_crop, model):
    """
    input img_crop: face crop before resize
    output: landmarks as flatten np.array
    """
    net_input = img_crop.copy()
    if img_crop.ndim == 3:
        net_input = cv2.cvtColor(net_input, cv2.COLOR_BGR2GRAY)
    net_input = cv2.resize(net_input, model.input_shape[1:-1][::-1])
    net_input = net_input.reshape(1, net_input.shape[0], net_input.shape[1], 1)
    lmarks = model.predict(net_input).flatten()

    if "2d" in model.name:
        lmarks = lmarks.reshape(-1, 2)
    if "3d" in model.name:
        lmarks = lmarks.reshape(-1, 3)
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
        print("\n>>> Output images are placed in ./croptest_out folder")
        
    def mean_pixel_std(self):
        return self.lmarks_all.std(0).mean()


def plot2D(frame, y_true=None, y_pred=None):
    h, w = frame.shape[:2]
    fig, ax = plt.subplots()
    plt.imshow(frame, cmap="gray")
    if y_true is not None:
        plt.scatter(y_true[:,0]*w, y_true[:, 1]*h, s=10, marker="*")
    if y_pred is not None:
        plt.scatter(y_pred[:,0]*w, y_pred[:, 1]*h, s=10)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot


def plot3D(frame, lmarks, return_array=False):
    def angle2axis(ang):
        n = int(abs(ang // 9))
        if ang < 0:
            s = " " * (10 - n) + "<" * n + "{:02.0f}".format(round(abs(ang), 0)) + " " * 10
        else:
            s = " " * 10 + "{:02.0f}".format(round(abs(ang), 0)) + ">" * n + " " * (10 - n)
        return s

    def head_pos(pleft, pright, ptop, pbottom):
        yaw = -math.atan((pright[2] - pleft[2]) / (pright[0] - pleft[0])) * 180 / math.pi
        pitch = math.atan((ptop[2] - pbottom[2]) / (ptop[1] - pbottom[1])) * 180 / math.pi
        return yaw, pitch

    lmarks = lmarks.copy()

    frame_height, frame_width = frame.shape[:2]
    lmarks[:, 0] = lmarks[:, 0] * frame_width
    lmarks[:, 1] = lmarks[:, 1] * frame_height
    lmarks[:, 2] = lmarks[:, 2] * (frame_width + frame_height) * 2
    yaw, pitch = head_pos(lmarks[4], lmarks[7], (lmarks[5] + lmarks[6]) / 2, lmarks[10])

    plt.rcParams['font.family'] = 'monospace'
    plt.subplots_adjust(left=0., bottom=0., right=1, top=1, wspace=0.2, hspace=0.2)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_zaxis().set_ticklabels([])

    xc, yc, zc = lmarks[:, 0], lmarks[:, 1], lmarks[:, 2]
    conts = [[11, 8, 10, 9, 11],
             [4, 5, 2, 1, 3, 6, 7],
             [5, 1, 6],
             [8, 2, 3, 9, 0, 8]]
    for cont in conts:
        ax.plot(xc[cont], yc[cont], zs=zc[cont], color="black", linewidth=2)

    ax.scatter(lmarks[:, 0], lmarks[:, 1], lmarks[:, 2])
    ax.set_xlim(frame_width, 0)
    ax.set_ylim(0, frame_height)
    ax.set_zlim(0, frame_width)

    ax.view_init(120, 90)
    ax.dist = 7
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(frame, cmap="gray")
    plt.scatter(lmarks[:, 0], lmarks[:, 1], s=10)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    plt.xlabel(angle2axis(yaw))
    plt.ylabel(angle2axis(pitch))
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)

    if return_array:
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.clf()
        plt.close(fig)
        return image_from_plot

    return fig


def cam(lmarks_predict_func, zoom=1):
    lmarks_ma = None
    with create_camera_reader(need_timestamps=True, mirror=True, delay=0) as camera_reader:
        for frame, cur_time in camera_reader:
            h, w = frame.shape[:2]
            cy, cx = h//2, w//2
            d = int(h/zoom)
            box = [cx-d//2, cy-d//2, cx+d//2, cy+d//2]

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            crop = frame[box[1]:box[3], box[0]:box[2]]

            lmarks = lmarks_predict_func(crop)
            if lmarks_ma is None:
                lmarks_ma = lmarks
            else:
                lmarks_ma = lmarks_ma*0.6 + lmarks*0.4
            if lmarks_ma.shape[1] == 2:
                picture2show = plot2D(frame=crop,
                                      y_true=None,
                                      y_pred=lmarks_ma.reshape((-1, 2)))
            elif lmarks_ma.shape[1] == 3:
                picture2show = plot3D(frame=crop,
                                      lmarks=lmarks_ma.reshape((-1, 3)),
                                      return_array=True)

            cv2.imshow("cam: for exit press <q>", picture2show)


if __name__ == "__main__":
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from argparse import ArgumentParser
    from tensorflow.keras.models import load_model
    from functools import partial

    parser = ArgumentParser()
    parser.add_argument("api", choices=["croptest", "cam"], type=str, help="available choices")
    parser.add_argument("-m", type=str, help="path to h5 model file")
    parser.add_argument("--img", type=str, help="path to image file")
    parser.add_argument("--zoom", type=float, help="scale bounding box, 1 - box equal frame height")
    args = parser.parse_args()

    model = load_model(args.m)
    print(model.name)
    print(model.summary())
    kw = partial(keras_wrapper, model=model)

    if args.api == "croptest":
        ct = CropTester(kw)
        if args.img is None:
            print("!!! --img argument must be defined")
        else:
            ct.test_image(args.img)
            print("\n>>> mean deviation of predictions, px:",  ct.mean_pixel_std())
    else:
        cam(kw, max(args.zoom, 1))
