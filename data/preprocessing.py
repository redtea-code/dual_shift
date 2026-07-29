"""Data normalization and table preprocessing utilities."""
import os
import re
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler


# ---- Image Normalization ----

def adaptive_normal(img):
    """Quantile-based normalization to [-1, 1]."""
    min_p = 0.001
    max_p = 0.999
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    imgArray = img
    imgPixel = imgArray[imgArray >= 0]
    imgPixel, _ = torch.sort(imgPixel)
    index = int(round(len(imgPixel) - 1) * min_p + 0.5)
    if index < 0:
        index = 0
    if index > (len(imgPixel) - 1):
        index = len(imgPixel) - 1
    value_min = imgPixel[index]

    index = int(round(len(imgPixel) - 1) * max_p + 0.5)
    if index < 0:
        index = 0
    if index > (len(imgPixel) - 1):
        index = len(imgPixel) - 1
    value_max = imgPixel[index]

    mean = (value_max + value_min) / 2.0
    stddev = (value_max - value_min) / 2.0
    imgArray = (imgArray - mean) / stddev
    imgArray[imgArray < -1] = -1.0
    imgArray[imgArray > 1] = 1.0
    return imgArray


def preprocess_one(pet_path):
    """Load and normalize a single NIfTI image."""
    img = nib.load(pet_path).get_fdata().astype(np.float32)
    # img = adaptive_normal(img)
    # img = img[:1, ...]
    return torch.from_numpy(img)


# ---- Table Preprocessing ----

def has_letters(string):
    if not isinstance(string, str):
        return False
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, string)
    return match is not None


def discovery_mix(df):
    str_columns = df.select_dtypes(include='object').columns
    mixed_columns = []
    for column in str_columns:
        if df[column].apply(has_letters).sum() > 0:
            mixed_columns.append(column)
    print("Mixed type columns:", mixed_columns)
    return mixed_columns


def prepare_table_n(mri_df, dataset='ADNI'):
    """Prepare tabular data with categorical/continuous split.
    ADNI: (TOTSCORE) VISDATE (MMSCORE) PTGENDER PTHAND PTMARRY PTEDUCAT PTNOTRT AGE_YEARS
    NACC: SEX HANDED MARISTAT EDUC (CDRSUM) AGE VISDATE label
    """
    drop_list = ['index','VISCODE2', "DIAGNOSIS"]
    info_list = ['PTID', 'VISDATE']
    if dataset == 'NACC':
        drop_list = ['label']
        info_list = ['PTID', 'VISDATE']
    table_info = mri_df[info_list]
    mri_df = mri_df.drop(drop_list + info_list, axis=1)
    mixed_columns = discovery_mix(mri_df)
    num_columns = [x for x in mri_df.columns if x not in mixed_columns]
    mri_df[mixed_columns] = mri_df[mixed_columns].fillna('NA').astype('category')
    num_cat = []
    for col in mri_df.columns:
        if col in mixed_columns:
            mri_df[col] = LabelEncoder().fit_transform(mri_df[col])
            num_cat.append(len(mri_df[col].unique()))
        else:
            mri_df[col] = pd.to_numeric(mri_df[col], errors='coerce')
            mri_df[col] = mri_df[col].fillna(0)
    sc = StandardScaler()
    sc.fit(mri_df[num_columns])
    mri_df[num_columns] = sc.transform(mri_df[num_columns])

    dfcats = mri_df[mixed_columns]
    df_categorical_encoded = dfcats
    dfconts = mri_df.drop(mixed_columns, axis=1)
    return_dict = {
        "info": table_info,
        "cate_x": df_categorical_encoded,
        "conti_x": dfconts,
        "num_cat": num_cat,
        "num_cont": len(num_columns)
    }
    return return_dict


def prepare_table_test(mri_df):
    """Prepare table for test dataset."""
    drop_list = ['Group', "Age"]
    info_list = ['Image Data ID', 'Subject']
    table_info = mri_df[info_list]
    mri_df = mri_df.drop(drop_list + info_list, axis=1)
    mixed_columns = discovery_mix(mri_df)
    num_columns = [x for x in mri_df.columns if x not in mixed_columns]
    mri_df[mixed_columns] = mri_df[mixed_columns].fillna('NA').astype('category')
    num_cat = []
    for col in mri_df.columns:
        if col in mixed_columns:
            mri_df[col] = LabelEncoder().fit_transform(mri_df[col])
            num_cat.append(len(mri_df[col].unique()))
        else:
            mri_df[col] = pd.to_numeric(mri_df[col], errors='coerce')
            mri_df[col] = mri_df[col].fillna(0)
    sc = StandardScaler()
    sc.fit(mri_df[num_columns])
    mri_df[num_columns] = sc.transform(mri_df[num_columns])

    dfcats = mri_df[mixed_columns]
    df_categorical_encoded = dfcats
    dfconts = mri_df.drop(mixed_columns, axis=1)
    return_dict = {
        "info": table_info,
        "cate_x": df_categorical_encoded,
        "conti_x": dfconts,
        "num_cat": num_cat,
        "num_cont": len(num_columns)
    }
    return return_dict
