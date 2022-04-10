import os
import cv2
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("data_dir", type=str)
args = parser.parse_args()

with open(os.path.join(args.data_dir, "annotations.txt"), "r") as f:
    line = f.readline()
    while len(line):
        fname, *vals = line.split()
        vals = list(map(int, vals))
        lmarks = np.array(vals)

        lmarks = lmarks.reshape(-1, 2)

        img = cv2.imread(os.path.join(args.data_dir, fname))
        for pt in lmarks:
            img = cv2.circle(img, tuple(pt), 1, (0, 0, 0), 3)
        cv2.imshow("face", img)
        k = cv2.waitKey(0)
        if k == 27:
            break
        line = f.readline()
