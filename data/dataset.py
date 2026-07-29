"""
V3 Dataset — multi-modal MRI classification.
dataset.py 本身与 V2 相同（修复了 super() bug 版本），
V3 的 CV 切分逻辑在 cv_splitter.py + trainer.py 中。
"""
import sys
import numpy as np
from tqdm import tqdm
from data.preprocessing import prepare_table_n, prepare_table_test

sys.path.append('./')
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils import data
import torch
import os
from glob import glob
import re
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Resized,
    Orientationd, ResampleToMatchd,
    EnsureTyped,
)
from utils.io_util import date_difference
from data.preprocessing import adaptive_normal, preprocess_one
import pandas as pd

aal_path = r'E:\GRETNA\Atlas\AAL116_1mm.nii'
template_path = r'E:\GRETNA\MNI152_T1_1mm_Brain.nii.gz'
target_size = (160, 160, 96)
target_spacing = (1.0, 1.0, 1.0)

register_transform = Compose([
    EnsureChannelFirstd(keys=["image", "template", 'aal']),
    Orientationd(keys=["image", 'template'], axcodes="RAS"),
    ResampleToMatchd(keys="image", key_dst="template", mode="bilinear"),
    Resized(keys=['image'], spatial_size=target_size, mode='bilinear'),
])
register_transform2 = Compose([
    EnsureChannelFirstd(keys=["template", 'aal']),
    Orientationd(keys=["aal", "template"], axcodes="RAS"),
    ResampleToMatchd(keys="aal", key_dst="template", mode="nearest"),
    Resized(keys=['aal'], spatial_size=target_size, mode='nearest'),
])


def read_nii(ni_path, desired_shape=(160, 160, 96)):
    img = nib.load(ni_path)
    data = img.get_fdata()
    desired_depth = desired_shape[2]
    desired_width = desired_shape[1]
    desired_height = desired_shape[0]
    current_depth = data.shape[2]
    current_width = data.shape[1]
    current_height = data.shape[0]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    return zoom(data, (height_factor, width_factor, depth_factor), order=1)


class PET_classify(data.Dataset):
    """Main dataset for multi-modal MRI classification."""

    def __init__(self, data_path, table_path=r'D:\1\5.AD\多分类数据集\merge.csv',
                days_threshold=90,
                 dataset='ADNI', save_load=False):
        super(PET_classify, self).__init__()
        self.dataset_name = dataset
        if save_load:
            self.PET_nii = glob(os.path.join(data_path, '*', 'MRI_brain_mni152_cropped_norm.nii.gz'))
            self.min_diff = days_threshold
            self.start_transformer = LoadImaged(keys=['image'])
            self.transformer = Compose([
                EnsureChannelFirstd(keys=['image']),
                ToTensord(keys=['image'])
            ])
            self.import_table = len(table_path)
            if self.import_table:
                self.table_df = pd.read_csv(table_path).dropna().reset_index()
                print(f"Num before filter: {len(self.PET_nii)}")
                to_remove = []
                for i, path in enumerate(self.PET_nii):
                    search_result = self.find_index(
                        mri_path=path.replace("\\", "/").split(r'/')[-2],
                        to_find_table=self.table_df
                    )
                    if search_result[0] == False:
                        to_remove.append(i)
                for i in reversed(to_remove):
                    self.PET_nii.pop(i)
                self.table_df = prepare_table_n(self.table_df, dataset=dataset)
                print(f"Num after filter: {len(self.PET_nii)}")
            self.save_data(data_path)
            print(f"save pt in {data_path}")
            self.PET_pt = glob(os.path.join(data_path, '*', '*.pt'))
        else:
            self.PET_pt = glob(os.path.join(data_path, '*', '*.pt'))
            self.import_table = len(table_path)
            if self.import_table:
                self.table_df = pd.read_csv(table_path).dropna().reset_index()
                print(f"Num after filter: {len(self.PET_pt)}")
                self.table_df = prepare_table_n(self.table_df, dataset=dataset)

    def find_row(self, ID, current_datetime, diagnosis, to_find_table):
        subset = to_find_table[(to_find_table['PTID'] == ID)]
        _min = self.min_diff if hasattr(self, 'min_diff') else 90
        min_index = -1
        for index, data in subset.iterrows():
            dateInCsv = data['VISDATE']
            diff = date_difference(dateInCsv, current_datetime)
            if _min > diff:
                _min = diff
                min_index = index
            if _min == 0:
                break
        if _min != (self.min_diff if hasattr(self, 'min_diff') else 90):
            return (True, min_index)
        else:
            print(
                f"找不到日期误差小于{getattr(self, 'min_diff', 90)}天 "
                f"diagnosis={diagnosis} (ID:{ID}, date:{current_datetime})！"
            )
            return (False, min_index)

    def find_index(self, mri_path, to_find_table=None):
        ID, date, diagnosis = mri_path.split('-')
        date = date.split('_')[0] + '-' + date.split('_')[1] + '-' + date.split('_')[2]
        status, min_index = self.find_row(ID, date, diagnosis, to_find_table)
        return (status, min_index)

    def __getitem__(self, index):
        pet_path = self.PET_pt[index]
        image = torch.load(pet_path, map_location="cpu")
        batch = {
            "image": image,
            "label": int(pet_path.split('\\')[-2][-1]),
        }
        if self.import_table:
            _, date_index = self.find_index(
                pet_path.replace("\\", "/").split('/')[-2], self.table_df['info']
            )
            batch['cate_x'] = torch.tensor(
                self.table_df['cate_x'].iloc[date_index].values, dtype=torch.int64
            )
            batch['conti_x'] = torch.tensor(
                self.table_df['conti_x'].iloc[date_index].values, dtype=torch.float32
            )
        return batch

    def save_data(self, save_dir):
        for pet_path in tqdm(self.PET_nii, desc='Saving .pt files'):
            out_path = os.path.join(
                save_dir,
                pet_path.replace('\\', '/').split('/')[-2],
                "MRI.pt"
            )
            if os.path.exists(out_path):
                continue
            tensor = preprocess_one(pet_path)
            torch.save(tensor, out_path)

    def __len__(self):
        return len(self.PET_pt)


class PET_classify_wo_table(data.Dataset):
    def __init__(self, data_path, table_path=r'D:\1\5.AD\多分类数据集\merge.csv',
                 desired_shape=(160, 160, 96)):
        super().__init__()
        self.PET_nii = glob(os.path.join(data_path, '*', '*.nii.gz'))
        self.start_transformer = LoadImaged(keys=['image', 'AAL'])
        if desired_shape == 'None':
            self.transformer = Compose([
                EnsureChannelFirstd(keys=["image", "AAL"]),
                Orientationd(keys=["image", "AAL"], axcodes="RAS"),
                EnsureTyped(keys=["image", "AAL"], track_meta=True),
                ResampleToMatchd(keys="image", key_dst="AAL", mode="bilinear"),
            ])
        else:
            self.transformer = Compose([
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=desired_shape),
                ToTensord(keys=['image'])
            ])
        self.import_table = len(table_path)
        if self.import_table:
            self.table_df = pd.read_csv(table_path).dropna().reset_index()
            print(f"Num before filter: {len(self.PET_nii)}")
            to_remove = []
            for i, path in enumerate(self.PET_nii):
                search_result = self.find_index(
                    PET_path=path.replace("\\", "/").split(r'/')[-2],
                    to_find_table=self.table_df
                )
                if search_result[0] == False:
                    to_remove.append(i)
            for i in reversed(to_remove):
                self.PET_nii.pop(i)
            self.table_df = prepare_table_test(self.table_df)
            print(f"Num after filter: {len(self.PET_nii)}")

    def find_row(self, ID, image_data, to_find_table):
        subset = to_find_table[(to_find_table['Subject'] == ID)]
        for index, data in subset.iterrows():
            if image_data == data['Image Data ID']:
                return (True, index)
        print(f"找不到对应数据(ID:{ID}, date:{image_data})！")
        return (False, -1)

    def __getitem__(self, index):
        pet_path = self.PET_nii[index]
        batch = self.start_transformer(dict(image=pet_path, AAL=aal_path))
        # batch['image'] = adaptive_normal(batch['image'])
        batch = self.transformer(batch)
        batch['image'] = batch["image"][:1, ...]
        batch['label'] = int(pet_path.split('\\')[-2].split('-')[-2])
        if self.import_table:
            _, date_index = self.find_index(
                pet_path.replace("\\", "/").split('/')[-2], self.table_df['info']
            )
            batch['cate_x'] = torch.tensor(
                self.table_df['cate_x'].iloc[date_index].values, dtype=torch.int64
            )
            batch['conti_x'] = torch.tensor(
                self.table_df['conti_x'].iloc[date_index].values, dtype=torch.float32
            )
        batch['name'] = pet_path.split('/')[-1]
        return batch

    def find_index(self, PET_path, to_find_table=None):
        ID, date, diagnosis, image_data = PET_path.split('-')
        status, min_index = self.find_row(ID, image_data, to_find_table)
        return (status, min_index)

    def __len__(self):
        return len(self.PET_nii)


class PET_classify_test(data.Dataset):
    """FIXED: use super().__init__() instead of wrong parent reference."""

    def __init__(self, data_path, table_path=r'D:\1\5.AD\多分类数据集\merge.csv',
                 desired_shape=(160, 160, 96), days_threshold=90, dataset='NACC'):
        super().__init__()  # FIXED — was super(PET_classify, self)
        self.PET_nii = glob(os.path.join(data_path, '*', '*.nii.gz'))
        self.min_diff = days_threshold
        self.start_transformer = LoadImaged(keys=['image', 'AAL'])
        if desired_shape == 'None':
            self.transformer = Compose([
                EnsureChannelFirstd(keys=["image", "AAL"]),
                Orientationd(keys=["image", "AAL"], axcodes="RAS"),
                EnsureTyped(keys=["image", "AAL"], track_meta=True),
                ResampleToMatchd(keys="image", key_dst="AAL", mode="bilinear"),
            ])
        else:
            self.transformer = Compose([
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=desired_shape),
                ToTensord(keys=['image'])
            ])
        self.import_table = len(table_path)
        if self.import_table:
            self.table_df = pd.read_csv(table_path).dropna().reset_index()
            print(f"Num before filter: {len(self.PET_nii)}")
            to_remove = []
            for i, path in enumerate(self.PET_nii):
                search_result = self.find_index(
                    mri_path=path.replace("\\", "/").split(r'/')[-2],
                    to_find_table=self.table_df
                )
                if search_result[0] == False:
                    to_remove.append(i)
            for i in reversed(to_remove):
                self.PET_nii.pop(i)
            self.table_df = prepare_table_n(self.table_df, dataset=dataset)
            print(f"Num after filter: {len(self.PET_nii)}")

    def find_row(self, ID, current_datetime, diagnosis, to_find_table):
        subset = to_find_table[(to_find_table['PTID'] == ID)]
        _min = self.min_diff
        min_index = -1
        for index, data in subset.iterrows():
            dateInCsv = data['VISDATE']
            diff = date_difference(dateInCsv, current_datetime)
            if _min > diff:
                _min = diff
                min_index = index
            if _min == 0:
                break
        if _min != self.min_diff:
            return (True, min_index)
        else:
            print(
                f"找不到日期误差小于{self.min_diff}天 "
                f"diagnosis={diagnosis} (ID:{ID}, date:{current_datetime})！"
            )
            return (False, min_index)

    def __getitem__(self, index):
        pet_path = self.PET_nii[index]
        batch = self.start_transformer(dict(image=pet_path, AAL=aal_path))
        batch['image'] = adaptive_normal(batch['image'])
        batch = self.transformer(batch)
        batch['image'] = batch["image"][:1, ...]
        batch['label'] = int(pet_path.split('\\')[-2][-1])
        if self.import_table:
            _, date_index = self.find_index(
                pet_path.replace("\\", "/").split('/')[-2], self.table_df['info']
            )
            batch['cate_x'] = torch.tensor(
                self.table_df['cate_x'].iloc[date_index].values, dtype=torch.int64
            )
            batch['conti_x'] = torch.tensor(
                self.table_df['conti_x'].iloc[date_index].values, dtype=torch.float32
            )
        batch['name'] = pet_path.split('/')[-1]
        return batch

    def find_index(self, mri_path, to_find_table=None):
        ID, date, diagnosis = mri_path.split('-')
        date = date.split('_')[0] + '-' + date.split('_')[1] + '-' + date.split('_')[2]
        status, min_index = self.find_row(ID, date, diagnosis, to_find_table)
        return (status, min_index)

    def __len__(self):
        return len(self.PET_nii)


if __name__ == "__main__":
    import sys, time
    from utils.io_util import see_mri_pet
    train_dataloader = PET_classify(
        r'E:\NACC\MRI', table_path=r'E:\NACC\MRI_mulclass3.csv'
    )
    start_time = time.time()
    batch = first(train_dataloader)
    end_time = time.time()
    print("Time: ", end_time - start_time)
    print("Shape: ", batch['image'].shape)
