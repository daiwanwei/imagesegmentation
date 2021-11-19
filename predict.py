import argparse
import os

import cv2
import numpy as np

from model.architecture.unet import unet
from model.backbone.resnet import get_resnet50_encoder
from utils.data_loader import get_image_arr

ROOT_DIR = os.path.abspath("./")
DEFAULT_PREDICT_DATA_DIR = os.path.join(ROOT_DIR, "data/predict")
DEFAULT_WEIGHT_PATH = os.path.join(ROOT_DIR, "weight/weight.ckpt.0")


def predict(model, img, input_size, output_size, n_classes):
    input_height, input_width = input_size
    output_height, output_width = output_size
    x = get_image_arr(img, input_height, input_width)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    return pr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict for image segmentation',
    )
    parser.add_argument(
        '--predict_dir', required=False, default=DEFAULT_PREDICT_DATA_DIR,
        metavar="/path/to/predictDir/", help='data directory for predict'
    )
    parser.add_argument(
        '--categories', required=False, default=2, metavar=2, help="categories", type=int,
    )
    parser.add_argument(
        '--weight_path', required=False, default=DEFAULT_WEIGHT_PATH,
        metavar="/path/to/weight.ckpt", help="model weight"
    )
    parser.add_argument(
        '--input_length', required=False, default=512, metavar=256, help="input_length", type=int,
    )

    parser.add_argument(
        '--input_width', required=False, default=512, metavar=256, help="input_width", type=int,
    )

    args = parser.parse_args()
    print("predict_dir: ", args.predict_dir)
    print("categories: ", args.categories)
    print("weight_path: ", args.weight_path)
    print("input_length: ", args.input_length)
    print("input_width: ", args.input_width)

    input_size = (args.input_length, args.input_width)
    n_classes = args.categories

    model, output_size = unet(n_classes, get_resnet50_encoder, input_size=input_size)

    if os.path.exists(args.weight_path):
        print("load weight successfully!")
        model.load_weights(args.weight_path)

    img_dir = os.path.join(args.predict_dir, 'img')
    if not os.path.exists(img_dir):
        raise Exception("img dir not found")

    mask_dir = os.path.join(args.predict_dir, 'mask')
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    names = os.listdir(img_dir)
    for name in names:
        if name.endswith(('.png', '.jpg')):
            img_path = os.path.join(img_dir, name)
            mask_path = os.path.join(mask_dir, name)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            img_height, img_width, _ = img.shape
            pr = predict(model, img, input_size, output_size, n_classes)
            mask = cv2.resize(pr, (img_height, img_width), interpolation=cv2.INTER_NEAREST).astype('uint8')
            cv2.imencode('.png', mask)[1].tofile(mask_path)
            print("predict mask successfully:{0}".format(name))
