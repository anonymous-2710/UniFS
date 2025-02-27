import numpy as np
import yaml
import argparse
import glob
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from Tools.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Testing Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    assert os.path.exists(config_file), "The configuration file does not exist."
    print(f"Using config file: {config_file}")
    print('*** Caution: Please use the coppied config file in the corresponding training log dir. ***')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("\n" + "*" * 10 + " Configuration " + "*" * 10)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("*" * 30 + "\n")
    ref_path = "Dataset/ref.nii.gz"
    ref_img = sitk.ReadImage(ref_path)

    experiment_name = config['training']['name']
    result_dir = os.path.join(config['test']['result_dir'], str(experiment_name))
    model_name = config['model']['target']
    model_params = config['model']['params']
    ckpt_save_path = os.path.join(config['training']['ckpt_save_dir'], str(experiment_name), "best_model.pth")
    batch_size = config['training']['batch_size']
    img_test_dir = config['test']['img_test_dir']
    dataset = config['test']['dataset']  # notice that it may be different from training dataset target
    wrapper = config['training']['wrapper']
    os.makedirs(result_dir, exist_ok=True)

    test_paths = glob.glob(os.path.join(img_test_dir, '**', '*.pkl'), recursive=True)  # 使用测试集路径
    test_ds = instantiate_from_config(model_name=dataset, dataset=test_paths)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(model_name, **model_params).to(device)
    try:
        model.load_state_dict(torch.load(ckpt_save_path))
    except RuntimeError as e:
        print(e)
        print('WARNING: Try to load ckpt with strict=false.')
        model.load_state_dict(torch.load(ckpt_save_path), strict=False)

    trainer = instantiate_from_config(model_name=wrapper, model=model, loss_fn=None)
    trainer.eval()
    total_psnr, total_ssim = [], []
    total_count = 0
    task_dict = {}
    task_list = os.listdir(img_test_dir)
    for task in task_list:
        test_paths = glob.glob(os.path.join(img_test_dir, task, '**', '*.pkl'), recursive=True)  # 使用测试集路径
        test_ds = instantiate_from_config(model_name=dataset, dataset=test_paths)
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=8, shuffle=False)

        task_psnr, task_ssim = [], []
        with torch.no_grad():
            for test_data in test_loader:  # 遍历测试集
                test_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in test_data.items()}
                meta_info = test_data['meta_index']
                t2_lrs = test_data['t2_lr']
                t1_hrs = test_data['t1_hr']
                preds, gts = trainer(test_data, return_preds=True)

                for i in range(preds.size(0)):  # 遍历当前批次的每个样本
                    pred = preds[i].squeeze().cpu().numpy()
                    gt = gts[i].squeeze().cpu().numpy()  # 200, 200 for compute SSIM
                    t2_lr = t2_lrs[i].squeeze().cpu().numpy()
                    t1_hr = t1_hrs[i].squeeze().cpu().numpy()

                    pred[pred < 0.01] = 0  # filter the little noise on background for better visualization
                    psnr = compute_psnr(gt, pred, data_range=1.0)
                    ssim = compute_ssim(gt, pred, data_range=1.0)
                    task_psnr.append(psnr)
                    task_ssim.append(ssim)
                    total_psnr.append(psnr)
                    total_ssim.append(ssim)
                    total_count += 1

                    img_name = meta_info[i]
                    print(f'{img_name} PSNR: {psnr:.4f} SSIM: {ssim:.4f}')
                    # For visualization
                    os.makedirs(os.path.join(result_dir, img_name), exist_ok=True)
                    save_image(pred[np.newaxis, ...], ref_img, os.path.join(result_dir, img_name, f't2_sp.nii.gz'))
                    save_image(gt[np.newaxis, ...], ref_img, os.path.join(result_dir, img_name, f't2_hr.nii.gz'))
                    save_image(t2_lr[np.newaxis, ...], ref_img, os.path.join(result_dir, img_name, f't2_lr.nii.gz'))
                    save_image(t1_hr[np.newaxis, ...], ref_img, os.path.join(result_dir, img_name, f't1_hr.nii.gz'))

            avg_psnr = np.mean(task_psnr)
            std_psnr = np.std(task_psnr)
            avg_ssim = np.mean(task_ssim)
            std_ssim = np.std(task_ssim)
            print(f"Test Average PSNR(std): {avg_psnr}({std_psnr}) Average SSIM(std): {avg_ssim}({std_ssim})")
            task_dict[task] = [avg_psnr, std_psnr, avg_ssim, std_ssim]

    print('#' * 20)
    for k, v in task_dict.items():
        print(f'Task {k}: Average PSNR(std) {round(v[0], ndigits=2)}({round(v[1], ndigits=2)}) '
              f'Average SSIM(std) {round(v[2], ndigits=3)}({round(v[3], ndigits=3)})')
    print(f'Total average PSNR(std): {round(np.mean(total_psnr), ndigits=2)}({round(np.std(total_psnr), ndigits=2)}) '
          f'Average SSIM(std): {round(np.mean(total_ssim), ndigits=3)}({round(np.std(total_ssim), ndigits=3)})')
