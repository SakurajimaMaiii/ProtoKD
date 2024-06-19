import os
import random
import numpy as np
import torch
import SimpleITK as sitk


def save_to_nii(npimg, path):
    itkimg = sitk.GetImageFromArray(npimg)
    sitk.WriteImage(itkimg, path)


def create_if_not(path):
    # create path if not exist
    if not os.path.exists(path):
        os.makedirs(path)


def set_random(seed_id=1234):
    # set random seed for reproduce
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)  # for cpu
    torch.cuda.manual_seed_all(seed_id)  # for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
