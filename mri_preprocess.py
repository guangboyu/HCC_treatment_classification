import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import tqdm
from config import data_paths
from nilearn.image import resample_img


def generate_separate_data():
    """
    Generate separate slice only contain non-zero mask
    """
    def separate_data(data_path, mask_path, save_path=None, type=None):
        """
        Transform 3D data to 2D like
        [160, 160, 15] -> [160, 160, 1]
        """
        print(f"Now processing {save_path} {type}")
        # read MRI sequence
        mri_raw = nib.load(data_path)
        mask_raw = nib.load(mask_path)
        mri_data = mri_raw.get_fdata()
        mask_data = mask_raw.get_fdata()
        # get non-zero mask slices
        result_mris = [mri_data[:, :, i] for i in range(mask_data.shape[2])
                        if np.any(mask_data[:, :, i])]
        result_mris = np.stack(result_mris, axis=2)
        result_masks = [mask_data[:, :, i] for i in range(mask_data.shape[2])
                        if np.any(mask_data[:, :, i])]
        result_masks = np.stack(result_masks, axis=2)
        print("result_mri shape", result_mris.shape)
        print("result_masks shape", result_masks.shape)
        # assert result_masks.shape[-1] == result_mris.shape[-1], "mri shape should equal to mask shape"
        # save separate slices with valid mask
        if save_path:
            for i in range(result_mris.shape[-1]):
                result_mri = result_mris[:, :, i]
                result_mri = nib.Nifti1Image(result_mri, mri_raw.affine, mri_raw.header)
                output_path = os.path.join(save_path, type + "_" + str(i) + ".nii.gz")
                print(output_path)
                nib.save(result_mri, output_path)
                print("saved to ", output_path)
            for i in range(result_masks.shape[-1]):
                result_mask = result_masks[:, :, i]
                result_mask = nib.Nifti1Image(result_mask, mask_raw.affine, mask_raw.header)
                output_mask_path = os.path.join(save_path, "MASK" + "_" + str(i) + ".nii.gz")
                nib.save(result_mask, output_mask_path)
                print(output_mask_path)
    
    for data_path in tqdm(data_paths):
        T1W_HR_path = glob.glob(os.path.join(data_path, "*T1W_HR.nii.gz"))[0]
        T2W_HR_path = glob.glob(os.path.join(data_path, "*T2W_HR.nii.gz"))[0]
        Mask_path = glob.glob(os.path.join(data_path, "*MASK2.nii.gz"))[0]
        separate_data(T1W_HR_path, Mask_path, save_path=data_path, type="T1W_HR_Separate")
        separate_data(T2W_HR_path, Mask_path, save_path=data_path, type="T2W_HR_Separate")
        
    print("Done")
    

# def resample_mismatch_mask():
#     """
#     correct mismatch mask in batch (not complete)
#     """
#     def correct(mri_path, mask_path):
#         # load data
#         image = nib.load(mri_path)
#         mask = nib.load(mask_path)
#         # get mask params
#         target_affine = mask.affine
#         print("affine", target_affine)
#         target_shape = mask.shape
#         # resample
#         resampled_image = resample_img(image, target_affine, target_shape)
#         # nib.save(resampled_image, mri_path)
#         print(f"{mri_path} saved")

#     mri_path_1 = "Data/Combination/B33R4_091922/DICOM_T1W_HR.nii.gz"
#     mask_path_1 = "Data/Combination/B33R4_091922/MASK2.nii.gz"
#     mri_path_2 = "Data/Combination/B35R2_101922/DICOM_T1W_HR.nii.gz"
#     mask_path_2 = "Data/Combination/B35R2_101922/MASK2.nii.gz"

#     correct(mri_path_1, mask_path_1)
#     correct(mri_path_2, mask_path_2)


if __name__ == '__main__':
    generate_separate_data()
    # resample_mismatch_mask()