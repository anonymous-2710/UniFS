import os
import pickle

import nibabel as nib
import numpy as np
from k_sampling import *
from utils import *

if __name__ == '__main__':
    sp = 'test'
    src_dir = f'root_src'  # BraTs 74:24:24
    tgt_dir = f'root_tar'
    visualize_dir = 'root_visual'
    ref_path = "../Dataset/ref.nii.gz"
    ref_img = sitk.ReadImage(ref_path)
    os.makedirs(visualize_dir, exist_ok=True)

    # BraTs: 240, 155, 240
    X = 120
    Y = np.arange(60, 100)
    Z = 120
    for dir in sorted(os.listdir(src_dir)):
        print('Processing {}'.format(dir))
        t1_mri_path = os.path.join(src_dir, dir, f'{dir}_t1.nii.gz')
        t2_mri_path = os.path.join(src_dir, dir, f'{dir}_t2.nii.gz')
        t1_mri = nib.load(t1_mri_path).get_fdata()
        t2_mri = nib.load(t2_mri_path).get_fdata()

        os.makedirs(os.path.join(tgt_dir, 'radial', dir), exist_ok=True)
        for y in Y:
            t1_slice = t1_mri[24:224, 24:224, y]
            t2_slice = t2_mri[24:224, 24:224, y]
            t1_hr = normalize_img(t1_slice)  # Normalize before fft
            t2_hr = normalize_img(t2_slice)
            # for better view on ITK-SNAP
            t1_hr = t1_hr.T
            t2_hr = t2_hr.T

            t2_lr_radial, k_sample, mask = degrade_radial(t2_hr, num_angles=32)

            t1_hr = t1_hr[np.newaxis, ...]
            t2_hr = t2_hr[np.newaxis, ...]

            path_dict = os.path.join(tgt_dir, 'radial', dir, f'{dir}_RAD_XZ{y}.pkl')
            data_dict = {'t1_hr': t1_hr, 't2_hr': t2_hr,
                         't2_lr': t2_lr_radial[np.newaxis, ...],
                         'k_space_cropped': numpy_complex_convert(k_sample),
                         'mask': mask[np.newaxis, ...]}
            with open(path_dict, 'wb') as f:
                pickle.dump(data_dict, f)

            # visualize
            # save_image(t1_hr, ref_img, os.path.join(visualize_dir, f'{dir}_T1HR_XZ{y}.nii.gz'))
            # save_image(t2_hr, ref_img, os.path.join(visualize_dir, f'{dir}_T2HR_XZ{y}.nii.gz'))
            # save_image(t2_lr_center[np.newaxis, ...], ref_img,
            #            os.path.join(visualize_dir, f'{dir}_T2LR_CEN_XZ{y}.nii.gz'))
            # save_image(t2_lr_random[np.newaxis, ...], ref_img,
            #            os.path.join(visualize_dir, f'{dir}_T2LR_RND_XZ{y}.nii.gz'))
            # save_image(t2_lr_equispace[np.newaxis, ...], ref_img,
            #            os.path.join(visualize_dir, f'{dir}_T2LR_EQS_XZ{y}.nii.gz'))
