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


if __name__ == '__main__':
    dataset_dir = "/mnt/lijl/Restormer/Dataset/Multix4/test/radial/"
    dst_dir = 'Dataset/Multix4_radial/checksum'
    # dataset_dir = "/mnt/lijl/Restormer/Dataset/FastMRI_Multix4/train/center/"
    # dst_dir = 'Dataset/FastMRI_Multix4/checksum'
    ref_path = "Dataset/ref.nii.gz"
    ref_img = sitk.ReadImage(ref_path)
    os.makedirs(dst_dir, exist_ok=True)
    batch_size = 4

    file_paths = glob.glob(os.path.join(dataset_dir, '**', '*.pkl'), recursive=True)
    # data_transforms = transforms.Compose([
    #     transforms.RandomRotation(15),  # 随机旋转
    #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #     # transforms.RandomVerticalFlip(),  # 随机垂直翻转
    #     # transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    # ])
    # dataset = DynamicMRIDataset(dataset=file_paths, transform=data_transforms)  # For train
    dataset = DynamicMRIDataset(dataset=file_paths)  # For train
    # dataset = MRIDataset(dataset=file_paths)  # For test
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 使用 next() 和 iter() 获取一个batch
    data_iter = iter(dataloader)
    data_dict = next(data_iter)
    # 遍历每张图像并保存
    for key in data_dict.keys():
        print(key)
        img_batch = data_dict[key]
        # if key == 'k_space_cropped':
        #     for i in range(batch_size):
        #         k_space_cropped = img_batch[i]
        #         k_space_complex = tensor_to_numpy_complex(k_space_cropped)[np.newaxis, ...]
        #         reconstructed_image = ifftshift(k_space_complex)
        #         reconstructed_image = np.abs(ifft2(reconstructed_image))
        #         k_space_visualize = np.log(np.abs(k_space_complex) + 1)
        #         save_image(reconstructed_image, ref_img, os.path.join(dst_dir, f'RecoverK_{i}.nii.gz'))
        #         save_image(k_space_visualize, ref_img, os.path.join(dst_dir, f'K_cropped_{i}.nii.gz'))
        #         print('k_space real min and max: ', np.real(k_space_complex).min(), np.real(k_space_complex).max())
        #         print('k_space imag min and max: ', np.imag(k_space_complex).min(), np.imag(k_space_complex).max())
        # elif key in ['t1_hr', 't2_hr', 't2_lr']:
        #     for i in range(batch_size):
        #         img = img_batch[i].numpy()  # 获取每张图像，形状是 (1, 200, 200)
        #         save_image(img, ref_img, os.path.join(dst_dir, f'{key}_{i}.nii.gz'))
        if key in ['t1_hr', 't2_hr', 't2_lr']:
            for i in range(batch_size):
                img = img_batch[i].numpy()  # 获取每张图像，形状是 (1, 200, 200)
                save_image(img, ref_img, os.path.join(dst_dir, f'{key}_{i}.nii.gz'))
