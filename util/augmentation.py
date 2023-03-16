import os
import torch
import torchvision.transforms as transforms

from augly.image import (EncodingQuality, OneOf, RandomBlur,
                         RandomEmojiOverlay, RandomPixelization,
                         RandomRotation, ShufflePixels)
from augly.image.functional import overlay_emoji, overlay_image, overlay_text
from augly.utils import pathmgr
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR, SMILEY_EMOJI_DIR

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image, ImageFilter
import pickle
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

class PseudoTransform:
    def __init__(self):
        self.aug_moderate = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]

        self.aug_hard = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.)),
            transforms.RandomGrayscale(p=0.25),
            transforms.RandomPerspective(p=0.35),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(value='random', p=0.25),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
        
        self.aug_weak = transforms.Compose(self.aug_moderate)
        self.aug_strong = transforms.Compose(self.aug_hard)

    def __call__(self, x):
        return (self.aug_weak(x), self.aug_strong(x))
    
