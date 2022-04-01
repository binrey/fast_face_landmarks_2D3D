import os
import cv2
import numpy as np

dbroot = "test_crop"
with open(os.path.join(dbroot, "annotations.txt"), "r") as f:
    line = f.readline()
    while len(line):
        fname, *vals = line.split()
        vals = list(map(int, vals))
        #box, lmarks = np.array(vals[:4]), np.array(vals[4:])
        lmarks = np.array(vals)

        #print(fname, box, lmarks)

        lmarks = lmarks.reshape(-1, 2)# + box[[0, 1]]

        img = cv2.imread(fname)

        #img = img[box[1]:box[3], box[0]:box[2]]
        #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,0), 7)
        for pt in lmarks:
            img = cv2.circle(img, tuple(pt), 1, (0, 0, 0), 3)
        #img = cv2.resize(img, (400, 500))
        cv2.imshow("face", img)
        cv2.waitKey()
        line = f.readline()
