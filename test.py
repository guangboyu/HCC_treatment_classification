from radiomics import featureextractor
import nibabel as nib
import glob
import os
from tqdm import tqdm
import pprint
import SimpleITK as sitk
import six
import time
import pandas as pd
import numpy as np

from config import data_paths


def check_dim():
    t1_mismatchs = 0
    t2_mismatchs = 0
    for data_path in data_paths:
        T1W_HR_path = glob.glob(os.path.join(data_path, "*T1W_HR.nii.gz"))[0]
        T2W_HR_path = glob.glob(os.path.join(data_path, "*T2W_HR.nii.gz"))[0]
        Mask_path = glob.glob(os.path.join(data_path, "*MASK2.nii.gz"))[0]
        t1_raw = nib.load(T1W_HR_path)
        t2_raw = nib.load(T2W_HR_path)
        mask_raw = nib.load(Mask_path)
        t1_data = t1_raw.get_fdata()
        t2_data = t2_raw.get_fdata()
        mask_data = mask_raw.get_fdata()
        if t1_data.shape != mask_data.shape:
            t1_mismatchs += 1
            print("t1 shape not equal: ", data_path)
            print(f"t1 shape {t1_data.shape}, mask shape{mask_data.shape}")
        if t2_data.shape != mask_data.shape:
            t2_mismatchs += 1
            print("t2 shape not equal: ", data_path)
    print("t1 mismatch: ", t1_mismatchs)
    print("t2 mismatch: ", t2_mismatchs)
    

def check_features():
    data_path = "T2_features.csv"
    df = pd.read_csv(data_path)
    print(df.shape)
    

def check_mask_label():
    data_path = 'Data/Control/B44R4_122022/MASK2.nii.gz'
    mask = nib.load(data_path).get_fdata()
    print(np.unique(mask))
    
    


if __name__ == '__main__':
    # check_dim()
    # check_features()
    check_mask_label()