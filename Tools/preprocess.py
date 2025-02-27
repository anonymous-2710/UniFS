import os
import numpy as np
import nibabel as nib
import pickle
from k_sampling import *
from utils import *

if __name__ == '__main__':
    sp = 'train'
    src_dir = f'/mnt/lijl/BraTs/{sp}/'  # BraTs 74:24:24
    tgt_dir = f'/mnt/lijl/Restormer/Dataset/Multix8/{sp}/'
    visualize_dir = '/mnt/lijl/Restormer/Dataset/Multix8/visualize'
    ref_path = "../Dataset/ref.nii.gz"
    ref_img = sitk.ReadImage(ref_path)
    os.makedirs(visualize_dir, exist_ok=True)

    # BraTs: 240, 155, 240
    Y = np.arange(60, 100)
    for dir in sorted(os.listdir(src_dir)):
        print('Processing {}'.format(dir))
        t1_mri_path = os.path.join(src_dir, dir, f'{dir}_t1.nii.gz')
        t2_mri_path = os.path.join(src_dir, dir, f'{dir}_t2.nii.gz')
        t1_mri = nib.load(t1_mri_path).get_fdata()
        t2_mri = nib.load(t2_mri_path).get_fdata()

        os.makedirs(os.path.join(tgt_dir, 'center', dir), exist_ok=True)
        os.makedirs(os.path.join(tgt_dir, 'random', dir), exist_ok=True)
        os.makedirs(os.path.join(tgt_dir, 'equispace', dir), exist_ok=True)
        for y in Y:
            t1_slice = t1_mri[24:224, 24:224, y]
            t2_slice = t2_mri[24:224, 24:224, y]
            t1_hr = normalize_img(t1_slice)  # Normalize before fft
            t2_hr = normalize_img(t2_slice)
            # for better view on ITK-SNAP
            t1_hr = t1_hr.T
            t2_hr = t2_hr.T

            t2_lr_center, k_sample1, mask1 = degrade_center(t2_hr, crop_factor=0.25)  # x4 downsampling
            t2_lr_random, k_sample2, mask2 = degrade_random(t2_hr, center_fraction=0.08, acceleration=4)  # x4
            t2_lr_equispace, k_sample3, mask3 = degrade_equispace(t2_hr, center_fraction=0.08, acceleration=4)  # x4
            # t2_lr_center, k_sample1, mask1 = degrade_center(t2_hr, crop_factor=0.125)  # x8 downsampling
            # t2_lr_random, k_sample2, mask2 = degrade_random(t2_hr, center_fraction=0.04, acceleration=8)  # x8
            # t2_lr_equispace, k_sample3, mask3 = degrade_equispace(t2_hr, center_fraction=0.04, acceleration=8)  # x8

            t1_hr = t1_hr[np.newaxis, ...]
            t2_hr = t2_hr[np.newaxis, ...]

            path_dict = os.path.join(tgt_dir, 'center', dir, f'{dir}_CEN_XZ{y}.pkl')
            data_dict = {'t1_hr': t1_hr, 't2_hr': t2_hr,
                         't2_lr': t2_lr_center[np.newaxis, ...],
                         'k_space_cropped': numpy_complex_convert(k_sample1),
                         'mask': mask1[np.newaxis, ...]}
            with open(path_dict, 'wb') as f:
                pickle.dump(data_dict, f)

            print('Saved {}'.format(path_dict))
            path_dict = os.path.join(tgt_dir, 'random', dir, f'{dir}_RND_XZ{y}.pkl')
            data_dict = {'t1_hr': t1_hr, 't2_hr': t2_hr,
                         't2_lr': t2_lr_random[np.newaxis, ...],
                         'k_space_cropped': numpy_complex_convert(k_sample2),
                         'mask': mask2[np.newaxis, ...]}
            with open(path_dict, 'wb') as f:
                pickle.dump(data_dict, f)

            print('Saved {}'.format(path_dict))
            path_dict = os.path.join(tgt_dir, 'equispace', dir, f'{dir}_EQS_XZ{y}.pkl')
            data_dict = {'t1_hr': t1_hr, 't2_hr': t2_hr,
                         't2_lr': t2_lr_equispace[np.newaxis, ...],
                         'k_space_cropped': numpy_complex_convert(k_sample3),
                         'mask': mask3[np.newaxis, ...]}
            with open(path_dict, 'wb') as f:
                pickle.dump(data_dict, f)
            print('Saved {}'.format(path_dict))

            # visualize
            # save_image(t1_hr, ref_img, os.path.join(visualize_dir, f'{dir}_T1HR_XZ{y}.nii.gz'))
            # save_image(t2_hr, ref_img, os.path.join(visualize_dir, f'{dir}_T2HR_XZ{y}.nii.gz'))
            # save_image(t2_lr_center[np.newaxis, ...], ref_img,
            #            os.path.join(visualize_dir, f'{dir}_T2LR_CEN_XZ{y}.nii.gz'))
            # save_image(t2_lr_random[np.newaxis, ...], ref_img,
            #            os.path.join(visualize_dir, f'{dir}_T2LR_RND_XZ{y}.nii.gz'))
            # save_image(t2_lr_equispace[np.newaxis, ...], ref_img,
            #            os.path.join(visualize_dir, f'{dir}_T2LR_EQS_XZ{y}.nii.gz'))
