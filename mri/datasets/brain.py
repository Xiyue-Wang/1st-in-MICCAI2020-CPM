

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.augmentations.spatial_transformations import augment_mirroring,augment_rot90,augment_zoom,augment_resize
import cv2
from skimage import transform

class BrainDataset_AO(Dataset):

    def __init__(self, img_list, sets,phase):
        # with open(img_list, 'r') as f:
        #     self.img_list = [line.strip() for line in f]
        # print("Processing {} datas".format(len(self.img_list)))
        #self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = phase


        subjects = open(img_list).read().splitlines()

        names = [sub.split(" ")[0].split('/')[-1] for sub in subjects]
        self.labels = [sub.split(" ")[-1] for sub in subjects]
        self.data_ids = [os.path.join(sub.split(" ")[0], name + '_') for sub, name in zip(subjects, names)]



        for i in range(len(self.labels)-1,-1,-1):
            if self.labels[i]== "G":
                del self.labels[i]
                del self.data_ids[i]





    # def __nii2tensorarray__(self, data):
    #     [z, y, x] = data.shape
    #     new_data = np.reshape(data, [1, z, y, x])
    #     new_data = new_data.astype("float32")
    #
    #     return new_data
    #
    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):

        if self.phase == "train":
            # read image and labels
            # ith_info = self.img_list[idx].split(" ")
            # img_name = os.path.join(self.root_dir, ith_info[0])
            # label_name = os.path.join(self.root_dir, ith_info[1])
            # assert os.path.isfile(img_name)
            # assert os.path.isfile(label_name)
            # img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            # assert img is not None
            # mask = nibabel.load(label_name)
            # assert mask is not None

            path = self.data_ids[idx]

            # path = self.data[idx]


            if self.labels[idx] == "A":
                labels = np.array([0.])
            else:
                labels = np.array([1.])

            modalities = ('flair', 't1ce', 't1', 't2', "seg_pro")

            img = np.stack([np.array(nibabel.load(path + modal + '.nii.gz').get_data(), dtype='float32', order='C')
                               for modal in modalities], -1)

            img = np.transpose(img, (3, 2, 1, 0))

            # final_img = nibabel.Nifti1Image(img[0], affine=np.eye(4))
            #
            # nibabel.save(final_img, str(idx) + "seg1.nii")

            # data processing
            img_array= self.__training_data_process__(img)

            # final_img = nibabel.Nifti1Image(img_array[0], affine=np.eye(4))
            #
            # nibabel.save(final_img, str(idx) + "seg2.nii")


            # 2 tensor array
            img_array =  img_array.astype("float32")#self.__nii2tensorarray__(img_array)
            labels = labels.astype("float32")#self.__nii2tensorarray__(mask_array)

            #assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, labels

        elif self.phase == "valid":
            # read image
            path = self.data_ids[idx]



            if self.labels[idx] == "A":
                labels = np.array([0.])
            else:
                labels = np.array([1.])

            modalities = ('flair', 't1ce', 't1', 't2', "seg_pro")

            img = np.stack([np.array(nibabel.load(path + modal + '.nii.gz').get_data(), dtype='float32', order='C')
                            for modal in modalities], -1)
            img = np.transpose(img, (3, 2, 1, 0))



            # data processing
            img_array = self.__valid_data_process__(img)

            # 2 tensor array
            img_array = img_array.astype("float32")  # self.__nii2tensorarray__(img_array)
            labels = labels.astype("float32")  # self.__nii2tensorarray__(mask_array)

            # assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, labels





    def __itensity_normalize_one_volume__(self,volume,flag):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std

        if flag=="train" and random.uniform(0., 1.) > 0.5:
            out_noise = np.random.normal(0, 0.1, size=volume.shape)
            out_noise[volume == 0] = 0

            out = out + out_noise

        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def crop_center(self, img, cropd, croph, cropw):
        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape
        stard = dim // 2 - (cropd // 2)
        starth = height // 2 - (croph // 2)
        startw = width // 2 - (cropw // 2)
        z = random.randint(-10, 10)
        y = random.randint(-10, 10)
        x = random.randint(-10, 10)

        return img[:, z + stard:z + stard + cropd, y + starth:y + starth + croph, x + startw:x + startw + cropw]

    def crop_center_valid(self, img, cropd, croph, cropw):
        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape
        stard = dim // 2 - (cropd // 2)
        starth = height // 2 - (croph // 2)
        startw = width // 2 - (cropw // 2)
        z = 0
        y = 0
        x = 0

        return img[:, z + stard:z + stard + cropd, y + starth:y + starth + croph, x + startw:x + startw + cropw]


    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = self.__random_center_crop__ (data, label)

        return data, label

    def expand_center(self, img, cropd, croph, cropw):

        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape
        stard = (cropd // 2) - dim // 2
        starth = (croph // 2) - height // 2
        startw = (cropw // 2) - width // 2
        new_img = np.zeros((img.shape[0],cropd, croph, cropw))
        new_img[:,stard:stard + dim, starth:starth + height, startw:startw + width] = img[:]
        return new_img


    def crop_mask(self, img, cropd, croph, cropw,flag):

        shape=img[0].shape

        img=self.expand_center(img,shape[0]+cropd//2+10,shape[1]+croph//2+10,shape[2]+cropw//2+10)
        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape

        mask = (img[4] >= 0.5).astype("uint8")
        target_indexs = np.where(mask > 0)
        [x2, y2, z2] = np.max(np.array(target_indexs), axis=1)
        [x1, y1, z1] = np.min(np.array(target_indexs), axis=1)


        #print("1",min_D,max_D,min_H max_H,min_W max_W)
        #print(x1,x2,y1,y2,z1,z2)
        if flag=="train":
            x_center = (x1 + x2) // 2 + random.randint(-5, 5)
            y_center = (y1 + y2) // 2 + random.randint(-5, 5)
            z_center = (z1 + z2) // 2 + random.randint(-5, 5)
        else:
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            z_center = (z1 + z2) // 2


        x1 = max([0, x_center - cropd // 2])
        y1 = max([0, y_center - croph // 2])
        z1 = max([0, z_center - cropw // 2])
        #
        # z2 = z1 + cropw
        # if z2 > width:
        #     z2 = width
        #     z1 = z2 - cropw
        #
        # y2 = y1 + croph
        # if y2 > height:
        #     y2 = height
        #     y1 = y2 - croph
        #
        # x2 = x1 + cropd
        # if x2 > dim:
        #     x2 = dim
        #     x1 = x2 - cropd

        return img[:, x1:x1+cropd, y1:y1+croph, z1:z1+cropw]



    def random_flip_dimensions(self,n_dimensions):
        axis = list()
        for dim in range(n_dimensions):
            if np.random.choice([True, False]):
                axis.append(dim)

        return axis

    def flip_image(self,image, axis):

        new_data = np.copy(image)
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)

        return new_data
    def resize(self,img,factor):
        #print(img.shape)

        if random.uniform(0., 1.) > 0.5:

            img = np.transpose(img, (0, 3, 2, 1))#240 240 155
            target_shape = (int(img.shape[1] * factor), int(img.shape[2] * factor))
            data = np.zeros((5, target_shape[0], target_shape[1], img.shape[3]))
            for i in range(5):
                data[i] = cv2.resize(img[i], target_shape)
            #print("1",data.shape)
            return np.transpose(data, (0, 3, 2, 1))
        else:
            return img

    def __training_data_process__(self, data):
        # crop data according net input size
        #data = data.get_data()
        #label = label.get_data()

        # drop out the invalid range
        #data, label = self.__drop_invalid_range__(data, label)



        # data processing

        img,_ = augment_mirroring(data,axes=(0,1,2))
        img,_=augment_rot90(img,sample_seg=None,num_rot=(1, 2, 3), axes=(1,2))
        factor = random.uniform(0.9, 1.1)
        img = self.resize(img,factor)


        #img = transform.rescale(img, factor)



        # crop data
        #data=self.crop_center(data,128,192,192)
        data = self.crop_mask(img,128, 192, 192,"train")
        #data, label = self.__crop_data__(data, label)

        # resize data
        #data = self.__resize_data__(data)
        #label = self.__resize_data__(label)

        # normalization datas
        for k in range(4):
            data[k] = self.__itensity_normalize_one_volume__(data[k],"train")


        return data

    def __valid_data_process__(self, data):
        # crop data according net input size
        # data = data.get_data()
        # label = label.get_data()

        # drop out the invalid range
        # data, label = self.__drop_invalid_range__(data, label)

        # crop data
        # data=self.crop_center(data,128,192,192)
        data = self.crop_mask(data, 128, 192, 192, "valid")
        # data, label = self.__crop_data__(data, label)

        # resize data
        # data = self.__resize_data__(data)
        # label = self.__resize_data__(label)

        # normalization datas
        for k in range(4):
            data[k] = self.__itensity_normalize_one_volume__(data[k],"valid")


        #print(data.shape)

        return data


    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data



class BrainDataset_G(Dataset):

    def __init__(self, img_list, sets,phase):
        # with open(img_list, 'r') as f:
        #     self.img_list = [line.strip() for line in f]
        # print("Processing {} datas".format(len(self.img_list)))
        #self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = phase


        subjects = open(img_list).read().splitlines()

        names = [sub.split(" ")[0].split('/')[-1] for sub in subjects]
        self.labels = [sub.split(" ")[-1] for sub in subjects]
        self.data_ids = [os.path.join(sub.split(" ")[0], name + '_') for sub, name in zip(subjects, names)]

        # if phase=="valid":
        #     subjects = open('/mnt/group-ai-medical/private/xiyiwu/data/CPM_2020/valid_all.txt').read().splitlines()
        #
        #     names = [sub.split(" ")[0].split('/')[-1] for sub in subjects]
        #     self.labels += [sub.split(" ")[-1] for sub in subjects]
        #     self.data_ids += [os.path.join(sub.split(" ")[0], name + '_') for sub, name in zip(subjects, names)]


    #
    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):

        if self.phase == "train":
            # read image and labels
            # ith_info = self.img_list[idx].split(" ")
            # img_name = os.path.join(self.root_dir, ith_info[0])
            # label_name = os.path.join(self.root_dir, ith_info[1])
            # assert os.path.isfile(img_name)
            # assert os.path.isfile(label_name)
            # img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            # assert img is not None
            # mask = nibabel.load(label_name)
            # assert mask is not None

            path = self.data_ids[idx]

            # path = self.data[idx]


            if self.labels[idx] == "G":
                labels = np.array([0.])
            else:
                labels = np.array([1.])

            modalities = ('flair', 't1ce', 't1', 't2', "seg_pro")

            img = np.stack([np.array(nibabel.load(path + modal + '.nii.gz').get_data(), dtype='float32', order='C')
                               for modal in modalities], -1)

            img = np.transpose(img, (3, 2, 1, 0))

            # final_img = nibabel.Nifti1Image(img[0], affine=np.eye(4))
            #
            # nibabel.save(final_img, str(idx) + "seg1.nii")

            # data processing
            img_array= self.__training_data_process__(img)

            # final_img = nibabel.Nifti1Image(img_array[0], affine=np.eye(4))
            #
            # nibabel.save(final_img, str(idx) + "seg2.nii")


            # 2 tensor array
            img_array =  img_array.astype("float32")#self.__nii2tensorarray__(img_array)
            labels = labels.astype("float32")#self.__nii2tensorarray__(mask_array)

            #assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, labels

        elif self.phase == "valid":
            # read image
            path = self.data_ids[idx]



            if self.labels[idx] == "G":
                labels = np.array([0.])
            else:
                labels = np.array([1.])

            modalities = ('flair', 't1ce', 't1', 't2', "seg_pro")

            img = np.stack([np.array(nibabel.load(path + modal + '.nii.gz').get_data(), dtype='float32', order='C')
                            for modal in modalities], -1)
            img = np.transpose(img, (3, 2, 1, 0))



            # data processing
            img_array = self.__valid_data_process__(img)

            # 2 tensor array
            img_array = img_array.astype("float32")  # self.__nii2tensorarray__(img_array)
            labels = labels.astype("float32")  # self.__nii2tensorarray__(mask_array)

            # assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            return img_array, labels





    def __itensity_normalize_one_volume__(self,volume,flag):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std

        if flag=="train" and random.uniform(0., 1.) > 0.5:
            out_noise = np.random.normal(0, 0.1, size=volume.shape)
            out_noise[volume == 0] = 0

            out = out + out_noise

        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def crop_center(self, img, cropd, croph, cropw):
        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape
        stard = dim // 2 - (cropd // 2)
        starth = height // 2 - (croph // 2)
        startw = width // 2 - (cropw // 2)
        z = random.randint(-10, 10)
        y = random.randint(-10, 10)
        x = random.randint(-10, 10)

        return img[:, z + stard:z + stard + cropd, y + starth:y + starth + croph, x + startw:x + startw + cropw]

    def crop_center_valid(self, img, cropd, croph, cropw):
        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape
        stard = dim // 2 - (cropd // 2)
        starth = height // 2 - (croph // 2)
        startw = width // 2 - (cropw // 2)
        z = 0
        y = 0
        x = 0

        return img[:, z + stard:z + stard + cropd, y + starth:y + starth + croph, x + startw:x + startw + cropw]


    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = self.__random_center_crop__ (data, label)

        return data, label

    def expand_center(self, img, cropd, croph, cropw):

        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape
        stard = (cropd // 2) - dim // 2
        starth = (croph // 2) - height // 2
        startw = (cropw // 2) - width // 2
        new_img = np.zeros((img.shape[0],cropd, croph, cropw))
        new_img[:,stard:stard + dim, starth:starth + height, startw:startw + width] = img[:]
        return new_img


    def crop_mask(self, img, cropd, croph, cropw,flag):

        shape=img[0].shape

        img=self.expand_center(img,shape[0]+cropd//2+10,shape[1]+croph//2+10,shape[2]+cropw//2+10)
        # for n_slice in range(img.shape[0]):
        dim, height, width = img[0].shape

        mask = (img[4] >= 0.5).astype("uint8")
        target_indexs = np.where(mask > 0)
        [x2, y2, z2] = np.max(np.array(target_indexs), axis=1)
        [x1, y1, z1] = np.min(np.array(target_indexs), axis=1)


        #print("1",min_D,max_D,min_H max_H,min_W max_W)
        #print(x1,x2,y1,y2,z1,z2)
        if flag=="train":
            x_center = (x1 + x2) // 2 + random.randint(-5, 5)
            y_center = (y1 + y2) // 2 + random.randint(-5, 5)
            z_center = (z1 + z2) // 2 + random.randint(-5, 5)
        else:
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            z_center = (z1 + z2) // 2


        x1 = max([0, x_center - cropd // 2])
        y1 = max([0, y_center - croph // 2])
        z1 = max([0, z_center - cropw // 2])
        #
        # z2 = z1 + cropw
        # if z2 > width:
        #     z2 = width
        #     z1 = z2 - cropw
        #
        # y2 = y1 + croph
        # if y2 > height:
        #     y2 = height
        #     y1 = y2 - croph
        #
        # x2 = x1 + cropd
        # if x2 > dim:
        #     x2 = dim
        #     x1 = x2 - cropd

        return img[:, x1:x1+cropd, y1:y1+croph, z1:z1+cropw]



    def random_flip_dimensions(self,n_dimensions):
        axis = list()
        for dim in range(n_dimensions):
            if np.random.choice([True, False]):
                axis.append(dim)

        return axis

    def flip_image(self,image, axis):

        new_data = np.copy(image)
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)

        return new_data
    def resize(self,img,factor):
        #print(img.shape)

        if random.uniform(0., 1.) > 0.5:

            img = np.transpose(img, (0, 3, 2, 1))#240 240 155
            target_shape = (int(img.shape[1] * factor), int(img.shape[2] * factor))
            data = np.zeros((5, target_shape[0], target_shape[1], img.shape[3]))
            for i in range(5):
                data[i] = cv2.resize(img[i], target_shape)
            #print("1",data.shape)
            return np.transpose(data, (0, 3, 2, 1))
        else:
            return img

    def __training_data_process__(self, data):
        # crop data according net input size
        #data = data.get_data()
        #label = label.get_data()

        # drop out the invalid range
        #data, label = self.__drop_invalid_range__(data, label)



        # data processing

        img,_ = augment_mirroring(data,axes=(0,1,2))
        img,_=augment_rot90(img,sample_seg=None,num_rot=(1, 2, 3), axes=(1,2))
        factor = random.uniform(0.9, 1.1)
        img = self.resize(img,factor)


        #img = transform.rescale(img, factor)



        # crop data
        #data=self.crop_center(data,128,192,192)
        data = self.crop_mask(img,128, 192, 192,"train")
        #data, label = self.__crop_data__(data, label)

        # resize data
        #data = self.__resize_data__(data)
        #label = self.__resize_data__(label)

        # normalization datas
        for k in range(4):
            data[k] = self.__itensity_normalize_one_volume__(data[k],"train")


        return data

    def __valid_data_process__(self, data):
        # crop data according net input size
        # data = data.get_data()
        # label = label.get_data()

        # drop out the invalid range
        # data, label = self.__drop_invalid_range__(data, label)

        # crop data
        # data=self.crop_center(data,128,192,192)
        data = self.crop_mask(data, 128, 192, 192, "valid")
        # data, label = self.__crop_data__(data, label)

        # resize data
        # data = self.__resize_data__(data)
        # label = self.__resize_data__(label)

        # normalization datas
        for k in range(4):
            data[k] = self.__itensity_normalize_one_volume__(data[k],"valid")


        #print(data.shape)

        return data


    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data







