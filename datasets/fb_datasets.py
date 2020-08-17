#!/usr/bin/env/ python
# -*- encoding: utf-8 -*-

"""
@Author    : Vy Bui
@Contact   : buithevybkdn@gmail.com
@File      : fb_datasets.py
@Time      : 06/27/2020 8:56 AM
@Desc      : customized a data set for full body segmentation
@License   : Bap Ventures
"""

import os
import random

import cv2.cv2 as cv2
import numpy as np
import torch
from torch.utils import data

from utils.transforms import get_affine_transform


class FBDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=None, scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        if crop_size is None:
            crop_size = [512, 512]
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.transform = transform
        self.dataset = dataset

        list_path = os.path.join(self.root, self.dataset + '_id.txt')
        train_list = [i_id.strip() for i_id in open(list_path)]

        self.train_list = train_list
        self.number_samples = len(self.train_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w,h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        train_item = self.train_list[index]

        im_path = os.path.join(self.root, self.dataset + '_images', train_item + '.jpg')
        parsing_anno_path = os.path.join(self.root, self.dataset + '_segmentations', train_item + '.png')

        print("im path: {}".format(im_path))
        print("parsing anno path: {}".format(parsing_anno_path))

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
            print("segmentation shape: {}".format(parsing_anno.shape))
            print("img shape: {}".format(im.shape))

            if self.dataset == 'train' or self.dataset == 'trainval':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    print("is flip")
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]
                    person_center[0] = im.shape[1] - person_center[0] - 1
                    right_idx = [5, 7]
                    left_idx = [6, 8]

                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input_im = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input_im = self.transform(input_im)

        meta = {
            'name': train_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset == 'val' or self.dataset == 'test':
            return input_im, meta

        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)
            # print("scale input_ims: {}/{}".format(input_im.shape,label_parsing.shape))
            return input_im, label_parsing, meta


class FbDataValSet(data.Dataset):
    def __init__(self, root, dataset='val', crop_size=None, transform=None,
                 flip=False):
        if crop_size is None:
            crop_size = [512, 512]
        self.root = root
        self.crop_size = crop_size
        self.transform = transform
        self.flip = flip
        self.dataset = dataset
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)

        list_path = os.path.join(self.root, self.dataset + '_id.txt')

        val_list = [i_id.strip() for i_id in open(list_path)]

        self.val_list = val_list
        self.number_samples = len(self.val_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        val_item = self.val_list[index]
        # Load training image
        im_path = os.path.join(self.root, self.dataset + '_images', val_item + '.jpg')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input_im = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        input_im = self.transform(input_im)
        flip_input_im = input_im.flip(dims=[-1])
        if self.flip:
            batch_input_im = torch.stack([input_im, flip_input_im])
        else:
            batch_input_im = input_im

        meta = {
            'name': val_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return batch_input_im, meta