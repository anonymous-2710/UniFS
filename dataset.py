import glob
import pickle
import os
import random

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from torchvision import transforms
from Tools.utils import *
from Tools.k_sampling import *


class MRIDataset(Dataset):  # degrade offline
    def __init__(self, dataset, transform=None):
        self.file_paths = dataset
        self.transform = transform
        self.cls = {'CEN': torch.tensor([0, 0, 1]), 'RND': torch.tensor([0, 1, 0]), 'EQS': torch.tensor([1, 0, 0])}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        data_dict = self._load_nii(self.file_paths[index])
        if self.transform:
            seed = random.randint(0, 99999999)
            for key in ['t1_hr', 't2_hr', 't2_lr']:  # 1, H, W
                # set seed to apply the same transform to all images in dict
                torch.manual_seed(seed)
                data_dict[key] = self.transform(data_dict[key])

        meta_index = self.file_paths[index].split('/')[-1].split('.')[0]
        data_dict['meta_index'] = meta_index
        # data_dict['degradation_class'] = self.cls[meta_index.split('_')[2]]
        return data_dict

    def _load_nii(self, file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        # to tensor
        for key in data_dict.keys():  # 1, H, W
            data_dict[key] = torch.tensor(data_dict[key], dtype=torch.float32)

        # Add channel ==> CHW
        # if img_tensor.ndimension() == 2:
        #     img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度 (C, H, W)

        return data_dict


class DynamicMRIDataset(Dataset):  # degrade online
    def __init__(self, dataset, transform=None):
        self.file_paths = dataset
        self.transform = transform
        self.function_list = [degrade_center, degrade_random, degrade_equispace]
        # self.cls = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        data_dict = self._load_nii(self.file_paths[index])
        if self.transform:
            seed = random.randint(0, 99999999)
            for key in ['t1_hr', 't2_hr']:  # 1, H, W
                # set seed to apply the same transform to all images in dict
                torch.manual_seed(seed)
                data_dict[key] = self.transform(data_dict[key])  # tensor (1,200,200)

        # choose degraded type randomly, output LR, k-space and mask
        t2_hr = data_dict['t2_hr'].squeeze(0).numpy()
        choice = random.randint(0, len(self.function_list) - 1)
        degrade_function = self.function_list[choice]
        # degradation_class = self.cls[choice]
        image_reconstructed, k_space_cropped, mask = degrade_function(t2_hr)
        k_space_cropped = numpy_complex_convert(k_space_cropped)  # split real and imag into 2 channels

        data_dict.update({
            't2_lr': torch.tensor(image_reconstructed, dtype=torch.float32).unsqueeze(0),
            'k_space_cropped': torch.tensor(k_space_cropped, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'meta_index': self.file_paths[index].split('/')[-1].split('.')[0],
            # 'degradation_class': degradation_class
        })

        return data_dict

    def _load_nii(self, file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        # to tensor
        for key in data_dict.keys():  # 1, H, W
            data_dict[key] = torch.tensor(data_dict[key], dtype=torch.float32)

        # Add channel ==> CHW
        # if img_tensor.ndimension() == 2:
        #     img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度 (C, H, W)

        return data_dict


class DynamicFastMRIDataset(Dataset):  # degrade online
    def __init__(self, dataset, transform=None):
        self.file_paths = dataset
        self.transform = transform
        self.function_list = [degrade_center, degrade_random, degrade_equispace]
        self.cls = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        data_dict = self._load_nii(self.file_paths[index])
        if self.transform:
            seed = random.randint(0, 99999999)
            for key in ['t1_hr', 't2_hr']:  # 1, H, W
                # set seed to apply the same transform to all images in dict
                torch.manual_seed(seed)
                data_dict[key] = self.transform(data_dict[key])  # tensor (1,200,200)

        # choose degraded type randomly, output LR, k-space and mask
        kspace = data_dict['kspace'].squeeze(0).numpy()
        choice = random.randint(0, len(self.function_list) - 1)
        degrade_function = self.function_list[choice]
        degradation_class = self.cls[choice]
        # Special for fastMRI here
        image_reconstructed, k_space_cropped, mask = degrade_function(kspace, input_kspace=True)
        image_reconstructed = process_fastMRI_knee(image_reconstructed)

        k_space_cropped = numpy_complex_convert(k_space_cropped)  # split real and imag into 2 channels

        data_dict.update({
            't2_lr': torch.tensor(image_reconstructed, dtype=torch.float32).unsqueeze(0),
            'k_space_cropped': torch.tensor(k_space_cropped, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'meta_index': self.file_paths[index].split('/')[-1].split('.')[0],
            'degradation_class': degradation_class
        })

        return data_dict

    def _load_nii(self, file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        # to tensor
        for key in data_dict.keys():  # 1, H, W
            data_dict[key] = torch.tensor(data_dict[key], dtype=torch.float32)

        # Add channel ==> CHW
        # if img_tensor.ndimension() == 2:
        #     img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度 (C, H, W)

        return data_dict


