# ProtoKD
This repo includes:  
1. PyTorch official implementation of __Prototype Knowledge Distillation for Medical Segmentation with Missing Modality__  
by __[Shuai Wang](https://scholar.google.com/citations?user=UbGMEyQAAAAJ&hl=en)__, [__Zipei Yan__](https://scholar.google.com/citations?user=JZvRMrcAAAAJ&hl=en&oi=ao), [__Daoan Zhang__](https://dwan.ch/), __Haining Wei__, __Zhongsen Li__, [__Rui Li__](https://scholar.google.com/citations?user=zTByNnsAAAAJ&hl=en&oi=ao).  
IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023).  
[__arXiv__](https://arxiv.org/abs/2303.09830), [__IEEEXplore__](https://ieeexplore.ieee.org/abstract/document/10095014)  
2. Paper collection of missing modality in medical image segmentation.

## Overview
![Methods](https://github.com/SakurajimaMaiii/ProtoKD/assets/83657651/2684dae8-241d-45cd-b264-b6d2592219fc)

## Dependencies
```
python==3.8.13
torch==1.12.0
numpy==1.22.3
medpy==0.4.0
simpleitk==2.2.1
tensorboard==2.12.0
tqdm==4.65.0
```
## Dataset
Please download BraTS2018 training set from [here](https://www.med.upenn.edu/sbia/brats2018.html).
For preprocessing, please change `path` (Line35) in `preprocess.py`, and
```
python preprocess.py
```
the preprocessed data will be organzied as:
```
--code
--data
  --brats2018  ##pre-processed data, npy format
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
If you want to save visualization results (`nii.gz` format, you can open it using [ITK-Snap](http://www.itksnap.org/pmwiki/pmwiki.php) or [3D-slicer](https://www.slicer.org/)), please set:
```
--save_vis
```
## Results
![results](https://github.com/SakurajimaMaiii/ProtoKD/assets/83657651/e5febfb8-8a3d-4b09-a69c-0bfb6fe0fc69)

## Models
We provide models for teacher, baseline T1, protokd T1 in [Google Drive](https://drive.google.com/drive/folders/1DhCBMn5Z002TzsfRwFzu_pvXjYc4BoCn?usp=sharing).

## Citation
If this code or ProtoKD is useful for your research, please consider citing our paper:
```
@inproceedings{Wang2023,
  year = {2023},
  month = jun,
  publisher = {{IEEE}},
  author = {Shuai Wang and Zipei Yan and Daoan Zhang and Haining Wei and Zhongsen Li and Rui Li},
  title = {Prototype Knowledge Distillation for Medical Segmentation with Missing Modality},
  booktitle = {{ICASSP} 2023 - 2023 {IEEE} International Conference on Acoustics,  Speech and Signal Processing ({ICASSP})}
}
```

## Contact
If you have any question, please contact bit.ybws@gmail.com
## 📜 Papers of missing modality in medical image analysis
We collected some papers about missing modality in medical image analysis, which may help people who are interested in this topic. Papers with public code are __highlighted__. If I miss some important papers, feel free to tell me.
* __Learning with Privileged Multimodal Knowledge for Unimodal Segmentation, [IEEE TMI 2022](https://ieeexplore.ieee.org/document/9567675), [code](https://github.com/cchen-cc/PMKL)__  
* __D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities, [IEEE TMI 2022](https://ieeexplore.ieee.org/document/9567675), [code](https://github.com/CityU-AIM-Group/D2Net)__ 
* __mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation, MICCAI 2022, [arXiv](https://arxiv.org/abs/2206.02425), [code](https://github.com/YaoZhang93/mmFormer)__
* __Robust Multimodal Brain Tumor Segmentation via Feature Disentanglement and Gated Fusion, MICCAI 2019, [arXiv](https://arxiv.org/abs/2002.09708), [code](https://github.com/cchen-cc/Robust-Mseg)__  
* __ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities, MICCAI 2021, [arXiv](https://arxiv.org/abs/2106.14591), [code](https://github.com/Wangyixinxin/ACN)__  
* __Hetero-Modal Variational Encoder-Decoder for Joint Modality Completion and Segmentation, MICCAI 2019, [arXiv](https://arxiv.org/abs/1907.11150), [code](https://github.com/ReubenDo/U-HVED)__  
* Knowledge distillation from multi-modal to mono-modal segmentation networks, MICCAI 2020, [arXiv](https://arxiv.org/abs/2106.09564)  
* HeMIS: Hetero-Modal Image Segmentation, MICCAI 2016, [arXiv](https://arxiv.org/abs/1607.05194)  
* Brain Tumor Segmentation on MRI with Missing Modalities, IPMI 2019, [arXiv](https://arxiv.org/abs/1904.07290)  
* Latent Correlation Representation Learning for Brain Tumor Segmentation With Missing MRI Modalities, [IEEE TIP 2021](https://ieeexplore.ieee.org/document/9399263)  
* SMU-Net: Style matching U-Net for brain tumor segmentation with missing modalities, [arXiv](https://arxiv.org/abs/2204.02961)  
* Learning Cross-Modality Representations From Multi-Modal Images, [IEEE TMI 2018](https://ieeexplore.ieee.org/document/8456579)  
* Multi-Domain Image Completion for Random Missing Input Data, [IEEE TMI 2020](https://ieeexplore.ieee.org/abstract/document/9302720)
* Medical Image Segmentation on MRI Images with Missing Modalities: A Review, [arXiv 2022](https://arxiv.org/abs/2203.06217)
* Disentangle First, Then Distill: A Unified Framework for Missing Modality Imputation and Alzheimer’s Disease Diagnosis, [IEEE TMI 2023](https://ieeexplore.ieee.org/abstract/document/10184044)
* __Discrepancy and Gradient-Guided Multi-modal Knowledge Distillation for Pathological Glioma Grading, [MICCAI 2022](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_61), [code](https://github.com/CityU-AIM-Group/MultiModal-learning)__
* Gradient modulated contrastive distillation of low-rank multi-modal knowledge for disease diagnosis, [Medical Image Analysis 2023](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001342)
* __M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities, [arXiv 2023](https://arxiv.org/abs/2303.05302), [code](https://github.com/ccarliu/m3ae)__
* __Fundus-Enhanced Disease-Aware Distillation Model for Retinal Disease Classification from OCT Images, MICCAI 2023 [arXiv](https://arxiv.org/abs/2308.00291), [code](https://github.com/xmed-lab/FDDM)__
* Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling, [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.pdf)
* __M$`^2`$FTrans: Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation, [IEEE JBHI 2023](https://ieeexplore.ieee.org/document/10288381), [code](https://github.com/Jun-Jie-Shi/M2FTrans).__
