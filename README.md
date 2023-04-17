# ProtoKD
Offical PyTorch implement of paper:Prototype Knowledge Distillation for Medical Segmentation with Missing Modality by Shuai Wang, Zipei Yan, Daoan Zhang, Haining Wei, Zhongsen Li, Rui Li. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023).
[arXiv](https://arxiv.org/abs/2303.09830)

## Dependencies
```
torch
numpy
medpy
SimpleITK
tensorboard
tqdm (only used in evaluate.py)
'''

## Dataset
Please download BraTS2018 training set from [here](http://braintumorsegmentation.org/).
For preprocessing, please change `path' and `outputs' in `preprocess.py', and
```
python preprocess.py
'''
## Training
First, train a teacher model
```
python pretrain.py --log_dir your_dir --data_dir your_data_dir
'''
For baseline (unimodal in paper),
```
python train_baseline.py --log_dir your_dir --data_dir your_data_dir --modality 0
'''
where modality=0,1,2,3 denotes using T1/T2/T1ce/Falir images for training.
For ProtoKD (our method) (you may need to change `teachermodel_path' and `data_dir' in `config.py')
```
python train_protokd.py --modality 0 --log_dir your_dir
'''
## Test
```
python evaluate.py --data_dir your_data_dir --model_path your_model_path --test_modality 0 --output_path your_out_path
'''
If you want to save visualization results, please set:
```
--save_vis
'''
