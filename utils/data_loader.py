import numpy as np
import cv2
import glob
import itertools
import os
from tqdm import tqdm
import math
from .image_augmentation import augment_seg
import random
import tensorflow as tf

IMAGE_ORDERING = 'channels_last'


def get_pairs_from_paths(img_dir, mask_dir):
    names = os.listdir(img_dir)
    dataset_num = len(names)
    path_pairs = []
    for name in names:
        if name.endswith(('.png', '.jpg')):
            img_path = os.path.join(img_dir, name)
            mask_path = os.path.join(mask_dir, name)
            path_pairs.append((img_path, mask_path))
    return path_pairs, dataset_num


def get_image_arr(img, width, height, img_norm="divide"):
    if img_norm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif img_norm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif img_norm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0
    return img


def get_mask_arr(mask, n_classes, width, height, no_reshape=False):
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask[:, :, 0]
    mask_labels = np.zeros((width, height, n_classes))
    for c in range(n_classes):
        mask_labels[:, :, c] = (mask == c).astype(int)
    if no_reshape:
        return mask_labels
    mask_labels = np.reshape(mask_labels, (width * height, n_classes))
    return mask_labels


def image_segmentation_generator(train_img_dir, train_mask_dir,
                                 batch_size, n_classes,
                                 input_size, output_size, do_augment=True):
    img_seg_pairs, dataset_num = get_pairs_from_paths(train_img_dir, train_mask_dir)
    random.shuffle(img_seg_pairs)
    return ImageSegmentationGenerator(img_seg_pairs, batch_size, n_classes, input_size, output_size,
                                      do_augment=True), dataset_num


class ImageSegmentationGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, n_classes, input_size, output_size, do_augment=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.input_height, self.input_width = input_size
        self.output_height, self.output_width = output_size
        self.do_augment = do_augment
        self.indexes = np.arange(len(self.dataset))
        self.shuffle = shuffle
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):

        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_dataset = [self.dataset[k] for k in batch_indexes]
        batch_x, batch_y = self.batch_generator(batch_dataset)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def batch_generator(self, batch_dataset):
        x = []
        y = []
        for img_path, mask_path in batch_dataset:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 1)
            if self.do_augment:
                img, mask[:, :, 0] = augment_seg(img, mask[:, :, 0])
            x.append(get_image_arr(img, self.input_width, self.input_height))
            y.append(get_mask_arr(mask, self.n_classes, self.output_width, self.output_height))
        return np.array(x), np.array(y)

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        batch_x, batch_y = self.__getitem__(self.n)
        self.n += 1
        return batch_x, batch_y
