import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os
import torch
import sys
import importlib


def onehot(y, num_classes):
    y_onehot = torch.zeros(y.size(0), num_classes).to(y.device)
    if len(y.size()) == 1:
        y_onehot = y_onehot.scatter_(1, y.unsqueeze(-1), 1)
    elif len(y.size()) == 2:
        y_onehot = y_onehot.scatter_(1, y, 1)
    else:
        raise ValueError("[onehot]: y should be in shape [B], or [B, C]")
    return y_onehot


def sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


def numpy_complex_convert(numpy_array):
    # restore real and imag part separately: 1,H,W ==> 2,H,W
    real_part = np.real(numpy_array)
    imag_part = np.imag(numpy_array)
    stacked_array = np.stack([real_part, imag_part], axis=0)

    return stacked_array


def tensor_to_numpy_complex(tensor):
    # Pay attention to the order
    real_part = tensor[0].numpy()
    imag_part = tensor[1].numpy()
    complex_ndarray = real_part + 1j * imag_part

    return complex_ndarray


def instantiate_from_config(model_name, **kwargs):
    # 动态导入模块和对象
    module_name, class_name = model_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    # print(f'Initialize model {model_name}...')
    # 创建对象实例
    return model_class(**kwargs)


def save_image(img, ref_img, output_path):
    # img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img = sitk.GetImageFromArray(img)
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, output_path)
    print(f'{output_path} saved successfully.')


def normalize_img(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    normalized_data = (image_data - min_val) / (max_val - min_val)  # [0,1]

    return normalized_data


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    pass
