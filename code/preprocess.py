"""
preprocess data (nii.gz) to npy
"""

import os
import argparse
import SimpleITK as sitk
import numpy as np


def read_nii(path):
    itkimg = sitk.ReadImage(path)
    npimg = sitk.GetArrayFromImage(itkimg)
    npimg = npimg.astype(np.float32)
    return npimg


def convert_label(gt):
    new_gt = (gt == 4) * 1 + (gt == 1) * 2 + (gt == 2) * 3
    return new_gt


def zscore_nonzero(img):
    # use non-zero ROI mean and std to normalize
    mask = img.copy()
    mask[img > 0] = 1
    mean = np.sum(mask * img) / np.sum(mask)
    std = np.sqrt(np.sum(mask * (img - mean) ** 2) / np.sum(mask))
    img = (img - mean) / std
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess BraTS2018 dataset')
    parser.add_argument('--input_path', type=str, 
                       default='../../Dataset/BraTS2018/brats2018',
                       help='Input path to BraTS2018 dataset')
    parser.add_argument('--output_path', type=str,
                       default='../data/brats2018/',
                       help='Output path for processed npy files')
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    
    for tumor_type in ["HGG", "LGG"]:
        type_path = os.path.join(input_path, tumor_type)
        all_subjects = os.listdir(type_path)
        
        for sub in all_subjects:
            sub_path = os.path.join(type_path, sub)
            
            t1_path = os.path.join(sub_path, f"{sub}_t1.nii.gz")
            t2_path = os.path.join(sub_path, f"{sub}_t2.nii.gz")
            t1ce_path = os.path.join(sub_path, f"{sub}_t1ce.nii.gz")
            flair_path = os.path.join(sub_path, f"{sub}_flair.nii.gz")
            seg_path = os.path.join(sub_path, f"{sub}_seg.nii.gz")
            
            t1_img = read_nii(t1_path)
            t2_img = read_nii(t2_path)
            t1ce_img = read_nii(t1ce_path)
            flair_img = read_nii(flair_path)
            seg = read_nii(seg_path)
            seg = seg.astype(np.int8)
            
            t1_img = zscore_nonzero(t1_img)
            t2_img = zscore_nonzero(t2_img)
            t1ce_img = zscore_nonzero(t1ce_img)
            flair_img = zscore_nonzero(flair_img)
            seg = convert_label(seg)
            
            data = np.stack([t1_img, t2_img, t1ce_img, flair_img, seg])  # 5*155*240*240
            data = data[:, 5:145, 24:216, 24:216]  # crop to 5*140*192*192
            
            output_file = os.path.join(output_path, f"{sub}.npy")
            np.save(output_file, data)
            print(f"{sub} saved to {output_file}")


if __name__ == "__main__":
    main()