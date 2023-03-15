# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import pandas as pd
import math
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, mode, args):
    if mode == 'train':
        is_train = True
    transform = build_transform(is_train, args)

    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    # dataset = datasets.ImageFolder(root, transform=transform)
    dataset = VscDataset(args.data_path, mode=mode, transform=transform, args=args)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class VscDataset(Dataset):
    def __init__(self, root_path, mode='train', transform=None, args=None):
        self.root_path = root_path
        self.mode = mode
        self.transform = transform
        self.args = args

        if mode == 'train':
            file_path = os.path.join(self.root_path, 'train/reference')
            anno_path = os.path.join(self.root_path, 'train/train_reference_metadata.csv')
        elif mode == 'val':
            query_path = os.path.join(self.root_path, 'train/query')
            file_path = os.path.join(self.root_path, 'train/reference')
            anno_path = os.path.join(self.root_path, 'train/train_descriptor_ground_truth.csv')
        else:
            NotImplementedError
        anno = pd.read_csv(anno_path)
        self.video_ids = list(anno['video_id'])
        # self.dataset_lengths = list(map(lambda x: math.ceil(x), list(anno['duration_sec'])))

        self.total_frames = []
        batch_bar = tqdm(total=len(self.video_ids), dynamic_ncols=True, leave=False, position=0, desc=f'Ready to set {self.mode} video list')
        for video_id in self.video_ids:
            video = os.path.join(file_path, video_id)
            frames = sorted(entry.name for entry in os.scandir(video) if entry.name.endswith('png'))
            for frame in frames:
                frame_dir = os.path.join(file_path, video_id, frame)
                self.total_frames.append(frame_dir)
            batch_bar.update()
        batch_bar.close()  
        print(f'Total frame number is {self.__len__}')                  

    def __getitem__(self, index):
        if self.mode == 'train':
            frame = PIL.Image.open(self.total_frames[index])
            frame = self.transform(frame)
        return frame

    def __len__(self):
        return len(self.total_frames)