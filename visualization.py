import argparse
import os

import numpy as np
import cv2

ROOT_DIR = os.path.abspath("./")
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "data/train")

class_colors = [(32, 11, 119), (100, 100, 150), (100, 60, 0), (142, 0, 0),
                (60, 20, 220), (0, 74, 111), (153, 153, 190), (81, 0, 81),
                (180, 165, 180), (230, 0, 0), (160, 170, 250), (131, 177, 244),
                (0, 0, 255), (250, 250, 250), (232, 35, 244)]


def visualization(img, mask, n_classes):
    mask_inv = (mask[:, :] == 0).astype('uint8')
    seg_img = np.zeros((mask.shape[0], mask.shape[1], 3))
    colors = class_colors
    for c in range(n_classes-1):
        seg_img[:, :, 0] += ((mask[:, :] == c+1) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c+1) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c+1) * (colors[c][2])).astype('uint8')
    image_mix = cv2.bitwise_and(img, img, mask=mask_inv) + seg_img
    return image_mix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict for image segmentation',
    )
    parser.add_argument(
        '--data_dir', required=False, default=DEFAULT_DATA_DIR,
        metavar="/path/to/dataDir/", help='data directory'
    )
    parser.add_argument(
        '--categories', required=False, default=2, metavar=2, help="categories", type=int,
    )

    args = parser.parse_args()
    print("data_dir: ", args.data_dir)
    print("categories: ", args.categories)

    img_dir = os.path.join(args.data_dir, 'img')
    if not os.path.exists(img_dir):
        raise Exception("img dir not found")

    mask_dir = os.path.join(args.data_dir, 'mask')
    if not os.path.exists(mask_dir):
        raise Exception("mask dir not found")

    visual_dir = os.path.join(args.data_dir, 'visual')
    if not os.path.exists(visual_dir):
        os.mkdir(visual_dir)

    names = os.listdir(img_dir)

    for name in names:
        if name.endswith(('.png', '.jpg')):
            img_path = os.path.join(img_dir, name)
            mask_path = os.path.join(mask_dir, name)
            visual_path = os.path.join(visual_dir, name)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 1)[:, :, 0]
            mix = visualization(img, mask, args.categories)
            cv2.imencode('.png', mix)[1].tofile(visual_path)
            print("visual successfully:{0}".format(name))

