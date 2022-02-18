


## Combined Radiology and Pathology Classification

### MICCAI 2020 Combined Radiology and Pathology Classification [Challenge](https://miccai.westus2.cloudapp.azure.com/competitions/1#learn_the_details) (1st place solution)

### Hardware
* 4*NVIDIA Tesla P40 GPU cards
* 32GB of RAM
### Pre-requisites:
   * torch >=1.3.0, nibabel, batchgenerators, efficientnet_pytorch

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

RESNET=False  #False is resnet, True is densenet
model.conv1 = nn.Conv3d(4,....)  #The input channel is 5 if the tumor segmentation region exists, otherwise it is 4



model = densenet.densenet121(first=5，..)  #The input channel is 5 if the tumor segmentation region exists, otherwise it is 4 
#datasets.brain.py
BrainDataset_AO,BrainDataset_G # AO dataset，G_dataset  
return img_array[:4], labels #The input channel is 5 if the tumor segmentation region exists, otherwise it is 4
```
  2.Trainning Oligodendroglioma/Lower grade astrocytoma

  After the first stage of training, the second stage of training needs to use the weights trained in the first stage to warm up
```
cd mri
python train_ao.py
```
#### WSI training

This is similar to the previous training

1. Trainning Glioblastoma/None Glioblastoma
```
cd pathology
python train_g.py
```

2. Trainning Oligodendroglioma/Lower grade astrocytoma


```
cd pathology
python train_ao.py
```





####Reference
This is code based on [MedicalNet](https://github.com/Tencent/MedicalNet)


### Citation
Please use below to cite this paper if you find our work useful in your research.





