import os
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from utils import normalize_img
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def generate_unified_heatmaps(image_pairs, output_dir):
    """
    生成统一色标的批量热力图
    参数：
    - image_pairs: 包含六个元组的列表，每个元组是(img1_path, img2_path)
    - output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 第一步：计算全局最大差异值
    global_max = 0
    all_diffs = []
    model_names = []  # 新增：存储模型名称

    for pair in image_pairs:
        img1_path, img2_path = pair
        # 新增：从路径中提取模型名称
        path_parts = img1_path.split('/')
        model_name = path_parts[-3]  # 根据你的路径结构调整索引位置
        model_names.append(model_name)

        nii1 = nib.load(img1_path)
        nii2 = nib.load(img2_path)

        data1 = np.squeeze(nii1.get_fdata()).T
        data2 = np.squeeze(nii2.get_fdata()).T

        diff = np.abs(data1 - data2)
        current_max = np.max(diff)

        if current_max > global_max:
            global_max = current_max

        all_diffs.append(diff)  # 保存差异数据供后续使用

        data1 = normalize_img(data1)
        data2 = normalize_img(data2)
        psnr = compute_psnr(data1, data2, data_range=1.0)
        ssim = compute_ssim(data1, data2, data_range=1.0)
        print(img1_path)
        print('PSNR: {:.2f}'.format(psnr))
        print('SSIM: {:.3f}'.format(ssim))

    # 第二步：生成单个热力图和组合图
    plt.figure(figsize=(18, 12))  # 组合图尺寸
    plt.suptitle("Unified Difference Heatmaps (Jet Colormap)", fontsize=16, y=0.95)

    for idx, (diff, pair, name) in enumerate(zip(all_diffs, image_pairs, model_names)):
        # 生成单个热力图
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        heatmap = ax.imshow(diff, cmap='jet', vmin=0, vmax=global_max)
        # plt.colorbar(heatmap, label='Difference Value')
        plt.axis('off')
        # plt.title(name)

        # 修改保存路径：使用模型名称作为文件名
        save_path = os.path.join(output_dir, f'{name}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        # 添加到组合图
        plt.subplot(2, 3, idx + 1)  # 2行3列布局
        plt.imshow(diff, cmap='jet', vmin=0, vmax=global_max)
        plt.title(name, fontsize=10)  # 修改标题显示模型名称
        plt.axis('off')

    # 添加统一颜色条
    cax = plt.axes([0.92, 0.15, 0.02, 0.7])  # 颜色条位置调整
    plt.colorbar(cax=cax, label='Difference Value')

    # 保存组合图
    combined_path = os.path.join(output_dir, 'combined_heatmaps.png')
    plt.savefig(combined_path, bbox_inches='tight', dpi=150)
    print(f"所有结果已保存至：{output_dir}")


if __name__ == "__main__":
    # src = 'BraTS2021_01356_RND_XZ79'
    # src = 'BraTS2021_01639_RND_XZ84'
    src = 'BraTS2021_01047_RND_XZ73'
    dir_list = [f'/mnt/lijl/Restormer/Results/MTrans/inferTs/MTrans_Multix4_0217/{src}/',
                f'/mnt/lijl/Restormer/Results/DCAMSR/inferTs/DCAMSR_Multix4_0217/{src}/',
                f'/mnt/lijl/Restormer/Results/FSMNet/inferTs/FSMNet_Multix4_0125/{src}/',
                f'/mnt/lijl/Restormer/Results/MINet/inferTs/MINet_Multix4_0217/{src}/',
                f'/mnt/lijl/Restormer/Results/tmp/MCVarNet_Multix4_0219/{src}/',
                f'/mnt/lijl/Restormer/Results/tmp/RCAN_Final_LR_Multix4_0216/{src}/']
    image_pairs = []
    for src_dir in dir_list:
        t2_hr_path = os.path.join(src_dir, 't2_hr.nii.gz')
        t2_sp_path = os.path.join(src_dir, 't2_sp.nii.gz')  # 修改输出文件名
        image_pairs.append((t2_hr_path, t2_sp_path))

    out_dir = '/mnt/lijl/Restormer/Results/visualization3/'
    os.makedirs(out_dir, exist_ok=True)
    generate_unified_heatmaps(image_pairs, out_dir)
    # for src_dir in dir_list:
    #     out_dir = os.path.join('/mnt/lijl/Restormer/Results/visualization/', src_dir.split('/')[-3])
    #     os.makedirs(out_dir, exist_ok=True)
    #     t2_hr_path = os.path.join(src_dir, 't2_hr.nii.gz')
    #     t2_sp_path = os.path.join(src_dir, 't2_sp.nii.gz')  # 修改输出文件名
    #     output_path = os.path.join(out_dir, 'diff.png')
    #     # visualize_diff(t2_hr_path, t2_sp_path, output_path)
    #     nifti_heatmap(t2_hr_path, t2_sp_path, output_path)
