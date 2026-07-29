"""File I/O and utility functions."""
import os
import time
import torch
import matplotlib.pyplot as plt
from shutil import copyfile, copy
from datetime import datetime
from torchvision.utils import make_grid
import sys


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_plot_data(epoch: int, predictions: torch.Tensor, targets: torch.Tensor, parent_dir: str):
    save_data = {
        'epoch': epoch,
        'predictions': predictions,
        'targets': targets,
    }
    torch.save(save_data, f'{parent_dir}/epoch_{epoch}_data.pth')


def date_difference(date1, date2, format1='%Y-%m-%d', format2='%Y-%m-%d'):
    # Normalize: NACC MRI folder dates use '_' (e.g. 2023_11_14)
    date1 = date1.replace('_', '-')
    date2 = date2.replace('_', '-')
    date_format1 = format1
    date_format2 = format2
    datetime_object1 = datetime.strptime(date1, date_format1)
    datetime_object2 = datetime.strptime(date2, date_format2)
    difference = abs(datetime_object2 - datetime_object1)
    return difference.days


def see_mri_pet(tensor_3d, normalize=True):
    tensor_3d = tensor_3d[0, 0, ...]
    tensor_3d = tensor_3d.permute(2, 0, 1)
    pic = make_grid(tensor_3d.unsqueeze(1))
    if normalize:
        pic = (pic + 1) / 2
    return pic


def plt_mri_pet(data, save_path):
    num_slices = data.shape[-1]
    num_rows = num_slices // 10 + 1
    num_cols = min(num_slices, 10)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for i in range(num_slices):
        row_idx = i // 10
        col_idx = i % 10
        axes[row_idx, col_idx].imshow(data[:, :, i], cmap='gray')
        axes[row_idx, col_idx].axis('off')
    for i in range(num_slices, num_rows * num_cols):
        row_idx = i // 10
        col_idx = i % 10
        fig.delaxes(axes[row_idx, col_idx])
    plt.savefig(save_path)


def copy_yaml_to_folder(yaml_file, folder):
    os.makedirs(folder, exist_ok=True)
    file_name = os.path.basename(yaml_file)
    copy(yaml_file, os.path.join(folder, file_name))


def copy_yaml_to_folder_auto(yaml_file, folder):
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%m%d%H%M%S")
    program_name_with_ext = os.path.basename(sys.argv[0])
    program_name, ext = os.path.splitext(program_name_with_ext)
    exp_dir = os.path.join(folder, os.path.basename('exp_' + str(formatted_time) + '_' + program_name))
    os.makedirs(exp_dir, exist_ok=True)
    file_name = os.path.basename(yaml_file)
    copy(yaml_file, os.path.join(exp_dir, file_name))
    return exp_dir


def write_config(config_path, save_path):
    copyfile(config_path, save_path)


def create_exp_dir(exp_type, model_name, dataset, fold_idx, base_dir, yaml_file,model_kwargs):
    """
    创建单层实验目录，命名规则：实验类型_模型名称_数据集_foldN
    例：adpc_adpc64_ADNI_fold1
    """
    stage_tag = model_kwargs.get('film_stages', model_kwargs.get('descripe', 'none'))
    dir_name = f"{exp_type}_{model_name}_{dataset}_fold{fold_idx}_{stage_tag}"
    exp_dir = os.path.join(base_dir, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    file_name = os.path.basename(yaml_file)
    copy(yaml_file, os.path.join(exp_dir, file_name))
    return exp_dir
