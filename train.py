import wandb
import yaml
import argparse
import glob
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from Tools.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config_file = args.config
    assert os.path.exists(config_file), "The configuration file does not exist."
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("\n" + "*" * 10 + " Configuration " + "*" * 10)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("*" * 30 + "\n")
    print('Please check k-sampling before starting the training script!')

    experiment_name = config['training']['name']
    log_dir = os.path.join(config['log']['log_dir'], str(experiment_name))
    ckpt_save_dir = os.path.join(config['training']['ckpt_save_dir'], str(experiment_name))
    os.makedirs(ckpt_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    destination = os.path.join(log_dir, 'config.yaml')
    shutil.copy(config_file, destination)
    print(f"Copied config file to: {destination}")

    model_name = config['model']['target']
    model_params = config['model']['params']
    img_train_dir = config['data']['img_train_dir']
    img_val_dir = config['data']['img_val_dir']
    batch_size = config['training']['batch_size']
    max_epoch = config['training']['max_epoch']
    val_interval = config['training']['val_interval']
    initial_lr = config['training']['initial_lr']
    loss_type = config['training']['loss_type']
    dataset = config['data']['dataset']
    wrapper = config['training']['wrapper']

    # sys.stdout = Logger(log_dir)  # std print log, conflict with wandb
    wandb.init(
        project=config['log']['project'],  # 项目名称
        name=experiment_name,  # 实验名称
        dir=log_dir,  # 保存目录
        config=config
    )

    train_paths = glob.glob(os.path.join(img_train_dir, '**', '*.pkl'), recursive=True)
    val_paths = glob.glob(os.path.join(img_val_dir, '**', '*.pkl'), recursive=True)
    data_transforms = transforms.Compose([
        transforms.RandomRotation(15),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
    ])
    train_ds = instantiate_from_config(model_name=dataset, dataset=train_paths, transform=data_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=8, shuffle=True)
    val_ds = instantiate_from_config(model_name=dataset, dataset=val_paths)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=8, shuffle=True)

    # create Net, Loss and Adam optimizer
    best_metric = -1
    best_metric_epoch = -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(model_name=model_name, **model_params).to(device)
    loss_function = instantiate_from_config(model_name=loss_type)
    trainer = instantiate_from_config(model_name=wrapper, model=model, loss_fn=loss_function)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_epoch)

    # start a typical PyTorch training
    for epoch in range(max_epoch):
        print("=" * 10)
        trainer.train()
        epoch_loss = 0
        step = 0
        epoch_len = len(train_ds) // train_loader.batch_size  # 计算总批次数
        pbar = tqdm(train_loader, total=epoch_len, desc=f'Epoch {epoch + 1}/{max_epoch}')
        for batch_data in pbar:
            step += 1
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}

            optimizer.zero_grad()
            loss = trainer(batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        wandb.log({"train_loss": epoch_loss}, step=epoch + 1)

        if (epoch + 1) % val_interval == 0:
            trainer.eval()
            total_psnr = 0.0
            total_ssim = 0.0
            count = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_data.items()}
                    preds, gts = trainer(val_data, return_preds=True)
                    for i in range(preds.size(0)):  # 遍历当前批次的每个样本
                        pred = preds[i].squeeze().cpu().numpy()
                        gt = gts[i].squeeze().cpu().numpy()
                        psnr = compute_psnr(gt, pred, data_range=1.0)
                        ssim = compute_ssim(gt, pred, data_range=1.0)  # set multi-channel=True for RGB
                        total_psnr += psnr
                        total_ssim += ssim  # SSIM 返回 tensor，需要取值
                        count += 1

                avg_psnr = total_psnr / count
                avg_ssim = total_ssim / count
                print(f"Current epoch {epoch + 1}: PSNR: {avg_psnr:.4f} SSIM: {avg_ssim:.4f}")
                wandb.log({"PSNR": avg_psnr, "SSIM": avg_ssim}, step=epoch + 1)

                metric = avg_psnr  # change it later
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(ckpt_save_dir, f'best_model.pth'))
                    print(f"Epoch {best_metric_epoch} save new best model: PSNR{avg_psnr:.2f} SSIM{avg_ssim:.2f}")
                    wandb.log({"Best_PSNR": avg_psnr, "Best_SSIM": avg_ssim}, step=epoch + 1)

    torch.save(model.state_dict(), os.path.join(ckpt_save_dir, f'last_epoch.pth'))
    print(f"train completed, best_metric PSNR: {best_metric:.4f} at epoch: {best_metric_epoch}")
    wandb.finish()
