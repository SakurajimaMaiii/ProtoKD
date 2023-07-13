# ProtoKD
This repo includes:  
1. PyTorch official implementation of __Prototype Knowledge Distillation for Medical Segmentation with Missing Modality__  
by Shuai Wang, Zipei Yan, Daoan Zhang, Haining Wei, Zhongsen Li, Rui Li.  
IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023).  
[arXiv](https://arxiv.org/abs/2303.09830), [IEEE](https://ieeexplore.ieee.org/abstract/document/10095014)  
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
## Papers of missing modality in medical image segmentation
We collected some papers about missing modality in medical image segmentation, this may help people who are interested in this topic.
* Learning with Privileged Multimodal Knowledge for Unimodal Segmentation, [IEEE TMI 2022](https://ieeexplore.ieee.org/document/9567675), [code](https://github.com/cchen-cc/PMKL)  
* D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities, [IEEE TMI 2022](https://ieeexplore.ieee.org/document/9567675), [code](https://github.com/CityU-AIM-Group/D2Net)  
* mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation, MICCAI 2022, [arXiv](https://arxiv.org/abs/2206.02425), [code](https://github.com/YaoZhang93/mmFormer)  
* Robust Multimodal Brain Tumor Segmentation via Feature Disentanglement and Gated Fusion, MICCAI 2019, [arXiv](https://arxiv.org/abs/2002.09708), [code](https://github.com/cchen-cc/Robust-Mseg)  
* ACN: Adversarial Co-training Network for Brain Tumor Segmentation with Missing Modalities, MICCAI 2021, [arXiv](https://arxiv.org/abs/2106.14591), [code](https://github.com/Wangyixinxin/ACN)  
* Hetero-Modal Variational Encoder-Decoder for Joint Modality Completion and Segmentation, MICCAI 2019, [arXiv](https://arxiv.org/abs/1907.11150), [code](https://github.com/ReubenDo/U-HVED)  
* Knowledge distillation from multi-modal to mono-modal segmentation networks, MICCAI 2020, [arXiv](https://arxiv.org/abs/2106.09564)  
* HeMIS: Hetero-Modal Image Segmentation, MICCAI 2016, [arXiv](https://arxiv.org/abs/1607.05194)  
* Brain Tumor Segmentation on MRI with Missing Modalities, IPMI 2019, [arXiv](https://arxiv.org/abs/1904.07290)  
* Latent Correlation Representation Learning for Brain Tumor Segmentation With Missing MRI Modalities, [IEEE TIP 2021](https://ieeexplore.ieee.org/document/9399263)  
* SMU-Net: Style matching U-Net for brain tumor segmentation with missing modalities, [arXiv](https://arxiv.org/abs/2204.02961)  
* Learning Cross-Modality Representations From Multi-Modal Images, [IEEE TMI 2018](https://ieeexplore.ieee.org/document/8456579)  
* Multi-Domain Image Completion for Random Missing Input Data, [IEEE TMI 2020](https://ieeexplore.ieee.org/abstract/document/9302720)
