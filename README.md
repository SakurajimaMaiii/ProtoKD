# ProtoKD
This repo includes:  
1. PyTorch official implementation of Prototype Knowledge Distillation for Medical Segmentation with Missing Modality  
by Shuai Wang, Zipei Yan, Daoan Zhang, Haining Wei, Zhongsen Li, Rui Li.  
IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023).  
[arXiv](https://arxiv.org/abs/2303.09830)  
2. Paper collection of missing modality in medical image segmentation.

## Dependencies

```
python
torch
numpy
medpy
SimpleITK
tensorboard
tqdm (only used in evaluate.py)
```

## Dataset
Please download BraTS2018 training set from [here](http://braintumorsegmentation.org/).
For preprocessing, please change `path` in `preprocess.py`, and
```
python preprocess.py
```
the preprocessed data will be organzied as:
```
--code
--data
  --brats2018
    --Brats18_2013_0_1.npy
  --train_list.txt
  --val_list.txt
  --test_list.txt
```
## Training
First, train a teacher model
```
python pretrain.py --log_dir ../log/teachermodel
```
For baseline (unimodal in paper),
```
python train_baseline.py --log_dir ../log/unimodal_modality0 --modality 0
```
where modality=0,1,2,3 denotes using T1/T2/T1ce/Falir images for training.  
For ProtoKD (our method)
```
python train_protokd.py --modality 0 --log_dir ../log/protokd_modality0
```
## Test
```
python evaluate.py --model_path ../log/protokd_modality0/model/best_model.pth --test_modality 0 --output_path protokd_modality0_outputs
```
If you want to save visualization results, please set:
```
--save_vis
```

## Citation
If this code or ProtoKD is useful for your research, please consider citing our paper:
```
@article{wang2023prototype,
  title={Prototype Knowledge Distillation for Medical Segmentation with Missing Modality},
  author={Wang, Shuai and Yan, Zipei and Zhang, Daoan and Wei, Haining and Li, Zhongsen and Li, Rui},
  journal={arXiv preprint arXiv:2303.09830},
  year={2023}
}
```
## Papers of missing modality in medical image segmentation
* Learning with Privileged Multimodal Knowledge for Unimodal Segmentation, [IEEE TMI 2022](https://ieeexplore.ieee.org/document/9567675), [code](https://github.com/cchen-cc/PMKL)  
* D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities, [IEEE TMI 2022](https://ieeexplore.ieee.org/document/9567675), [code](https://github.com/CityU-AIM-Group/D2Net)  
* mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation, MICCAI 2022, [arXiv](https://arxiv.org/abs/2206.02425), [code](https://github.com/YaoZhang93/mmFormer)  
* ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities, MICCAI 2021, [arXiv](https://arxiv.org/abs/2106.14591), [code](https://github.com/Wangyixinxin/ACN)  
* Brain Tumor Segmentation on MRI with Missing Modalities, IPMI 2019, [arXiv](https://arxiv.org/abs/1904.07290)  
* Knowledge distillation from multi-modal to mono-modal segmentation networks, MICCAI 2020, [arXiv](https://arxiv.org/abs/2106.09564)  
* 
