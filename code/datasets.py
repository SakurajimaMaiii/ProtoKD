import random

import numpy as np
import torch
from torch.utils.data import Dataset



def random_crop(img, crop_size=(80, 80, 80)):
    # img [c,d,h,w]
    _, d, h, w = img.shape
    pd = np.random.randint(d - crop_size[0])
    ph = np.random.randint(h - crop_size[1])
    pw = np.random.randint(w - crop_size[2])
    patch = img[
        :, pd : pd + crop_size[0], ph : ph + crop_size[1], pw : pw + crop_size[2]
    ]
    return patch


def random_flip(data, p=0.5):
    # [C,D,H,W]
    if random.random() < p:
        data = np.flip(data, axis=1)
    if random.random() < p:
        data = np.flip(data, axis=2)
    if random.random() < p:
        data = np.flip(data, axis=3)
    return data


def random_scale_one_channel(data, p=0.5):
    if random.random() < p:
        scale = random.uniform(0.9, 1.1)
        data = data * scale
    return data


def random_scale(data, c=4, p=0.5):
    for i in range(c):
        data[i] = random_scale_one_channel(data[i], p)
    return data


class BraTS(Dataset):
    def __init__(
        self, base_dir="../data", crop_size=(80, 80, 80), flip=True, scale=True
    ):
        # just for train
        # base_dir  dataset path of brats dataset e.g. '../data'
        # student_modality: must be list e.g.[0] [1,2] or None(pre-train of teacher)
        imglist = []
        f = open(base_dir + "/train_list.txt", "r")
        lines = f.readlines()
        for ll in lines:
            imglist.append(ll.replace("\n", ""))
        f.close()
        self.imglist = [base_dir + "/brats2018/" + x + ".npy" for x in imglist]

        self.crop_size = crop_size
        self.flip = flip
        self.scale = scale

    def __getitem__(self, index):
        data = np.load(self.imglist[index])
        data = random_crop(data, self.crop_size)
        if self.flip:
            data = random_flip(data)
        label = data[4:]
        image = data[0:4]
        if self.scale:
            image = random_scale(image)
        image = torch.from_numpy(image.copy())
        label = torch.from_numpy(label.copy())
        return image, label

    def __len__(self):
        return len(self.imglist)
