import os
import pickle

import PIL

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


##TODO: Modify this function (img_size)
def get_img_trans(phase, image_size=291):
    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    if phase == "train":
        return A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(),
                normalize,
                ToTensor(),
            ]
        )
    elif phase in ["test", "val"]:
        return A.Compose(
            [
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                normalize,
                ToTensor(),
            ]
        )
    else:
        raise KeyError
