import os
import cv2
import numpy as np
import shutil
from glob import glob


def get_kpts(file_path):
    num_kpts = 68
    kpts = []
    with open(file_path, 'r') as f:
        ln = f.readline()
        while not ln.startswith('n_points'):
            ln = f.readline()
        num_pts = ln.split(':')[1]
        num_pts = num_pts.strip()
        # checking for the number of keypoints
        if float(num_pts) != num_kpts:
            print("encountered file with less than %f keypoints in %s" %(num_kpts, file_path))
            return None

        # skipping the line with '{'
        ln = f.readline()
        ln = f.readline()
        while not ln.startswith('}'):
            vals = ln.split()[:2]
            vals = list(map(lambda x: round(float(x), 0), vals))
            kpts.append(vals)
            ln = f.readline()
    return np.array(kpts, dtype=np.int32)


def calc_crop_box(gt_points, ratio):
    # crop face box
    x_min, y_min = gt_points.min(0)
    x_max, y_max = gt_points.max(0)
    w, h = x_max - x_min, y_max - y_min
    # w = h = min(w, h)
    x_new = max(0, x_min - w * ratio)
    y_new = max(0, y_min - h * ratio)
    w_new = w * (1 + 2 * ratio)
    h_new = h * (1 + 2 * ratio)
    box = np.array([x_new, y_new, x_new + w_new, y_new + h_new], dtype=np.int32)
    return box


def main(db_sourse, db_target, max_size, debug = False):
    shutil.rmtree(db_target, ignore_errors=True)
    os.makedirs(db_target, exist_ok=True)
    annfile_new = open(os.path.join(db_target, "annotations.txt"), "w")

    ann_files = []
    for db in os.listdir(db_sourse):
        print(f">>> Looking for pts files in {db}...")
        search_dir = os.path.join(db_sourse, db)
        if os.path.isdir(search_dir):# and search_dir == "test/lfpw":
            ann_files += sorted(glob(os.path.join(search_dir, "*.pts")))
    print(f">>> Read {len(ann_files)} files...\n\n")
    for i, ann_file in enumerate(ann_files):
        lmarks = get_kpts(ann_file)
        fname = ann_file.split('.')[0] + '.jpg'
        if not os.path.isfile(fname):
            fname = ann_file.split('.')[0] + '.png'

        print(f">>> {fname} -> ", end="")
        box = calc_crop_box(lmarks, 0.2)
        img = cv2.imread(fname)

        box[2] = min(box[2], img.shape[1])
        box[3] = min(box[3], img.shape[0])

        img = img[box[1]:box[3], box[0]:box[2]]
        if img.shape[0] > max_size:
            scale = max_size/img.shape[0]
            img = cv2.resize(img, (int(img.shape[1]*scale), max_size))
        if img.shape[1] > max_size:
            scale = max_size/img.shape[1]
            img = cv2.resize(img, (max_size, int(img.shape[0]*scale)))
        lmarks = lmarks.reshape(-1, 2)
        lmarks[:, 0] = (lmarks[:, 0] - box[0])/(box[2] - box[0])*img.shape[1]
        lmarks[:, 1] = (lmarks[:, 1] - box[1])/(box[3] - box[1])*img.shape[0]
        if debug:
            for pt in lmarks:
                img = cv2.circle(img, tuple(pt), 1, (0, 0, 0), 3)
            cv2.imshow("face", img)
            k = cv2.waitKey(0)
            if k == 27:
                break


        fname_new = "{:06d}.png".format(i)
        inner_dir = os.path.join(*os.path.normpath(fname).split(os.sep)[1:-1])
        dir2save = os.path.join(db_target, inner_dir)
        os.makedirs(dir2save, exist_ok=True)
        cv2.imwrite(os.path.join(dir2save, fname_new), img)

        annfile_new.write(os.path.join(inner_dir, fname_new))
        for pt in lmarks:
            annfile_new.write(" {} {}".format(pt[0], pt[1]))

        annfile_new.write("\n")
        print(os.path.join(inner_dir, fname_new))
    annfile_new.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("db_sourse")
    parser.add_argument("db_target")
    parser.add_argument("--max_size", type=int, default=256)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()


    main(db_sourse = args.db_sourse, db_target = args.db_target, max_size = args.max_size, debug = args.debug)