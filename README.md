


## Combined Radiology and Pathology Classification

### MICCAI 2020 Combined Radiology and Pathology Classification [Challenge](https://miccai.westus2.cloudapp.azure.com/competitions/1#learn_the_details) (1st place solution)

### Hardware
* 4*NVIDIA Tesla P40 GPU cards
* 32GB of RAM
### Pre-requisites:
   * torch >=1.3.0, nibabel, batchgenerators

### Usage
#### Preparation
  1. Data Preparation
  * Download challenge [data](https://miccai.westus2.cloudapp.azure.com/competitions/1#participate)
  2. Training Splits (spilt/train1.txt)

#### MRI training
  1. Trainning Glioblastoma/None Glioblastoma(pretrain 3d medical [Weights](https://drive.google.com/file/d/1BHbJ5JCh6IP5t4pdPT1eXkmT79e8DKgK/view?usp=sharing) )
```
cd mri
python train_g.py

RESNET=False  #False用resnet, True用densenet
model.conv1 = nn.Conv3d(4,....)  #输入通道数 用mask为5，否则4 



model = densenet.densenet121(first=5，..)  #输入通道数 用mask为5，否则4 
#datasets.brain.py
BrainDataset_AO,BrainDataset_G #分AO类，分G类的数据集读取  
return img_array[:4], labels #这里用来选择返回4通道或者带mask的5通道输入
```
  2.Trainning Oligodendroglioma/Lower grade astrocytoma
  After the first stage of training, the second stage of training needs to use the weights trained in the first stage to warm up
```
cd mri
python train_ao.py
```
#### WSI training



Reference
This is code based on [MedicalNet](https://github.com/Tencent/MedicalNet)


### Citation
Please use below to cite this paper if you find our work useful in your research.





# 1st-in-MICCAI2020-CPM
