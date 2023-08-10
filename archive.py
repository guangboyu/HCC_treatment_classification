# store functions don't need

# def extract_T2_all():
#     """
#     extract [160, 160, 15] like
#     """
#     params = 'Params.yaml'
#     extractor = featureextractor.RadiomicsFeatureExtractor(params)
#     # print('Extraction parameters:\n\t', extractor.settings)
#     # print('Enabled filters:\n\t', extractor.enabledImagetypes)
#     # print('Enabled features:\n\t', extractor.enabledFeatures)
#     start_time = time.time()
#     count = 0
#     errors = 0
#     features = pd.DataFrame()
#     for data_path in tqdm(data_paths):
#         if not glob.glob(os.path.join(data_path, "*T2W_HR.nii.gz")):
#             break
#         T2W_HR_path = glob.glob(os.path.join(data_path, "*T2W_HR.nii.gz"))[0]
#         Mask_path = glob.glob(os.path.join(data_path, "MASK2.nii.gz"))[0]
#         try:
#             result = extractor.execute(T2W_HR_path, Mask_path)
#             feature = pd.DataFrame([result])
#             features = pd.concat([features, feature])
#         except Exception as error:
#             print("T2 Exception occurs: ", error)
#             print("T2 error path: ", data_path)
#             errors += 1
#             continue
#         feature_size = len(result)
#         count += 1
#         # for key, val in six.iteritems(result):
#         #     pass
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Elapsed time: {elapsed_time:.6f} seconds")
#     print("T2 feature size: ", feature_size)
#     print("T2 sample size: ", count)
#     print("T2 error size: ", errors)
#     features.to_csv("T2_features.csv")


# def extract_T1_all():
#     """
#     extract [160, 160, 15] like
#     """
#     params = 'Params.yaml'
#     extractor = featureextractor.RadiomicsFeatureExtractor(params)
#     # print('Extraction parameters:\n\t', extractor.settings)
#     # print('Enabled filters:\n\t', extractor.enabledImagetypes)
#     # print('Enabled features:\n\t', extractor.enabledFeatures)
#     start_time = time.time()
#     count = 0
#     errors = 0
#     for data_path in tqdm(data_paths):
#         if not glob.glob(os.path.join(data_path, "*T1W_HR.nii.gz")):
#             break
#         T2W_HR_path = glob.glob(os.path.join(data_path, "*T1W_HR.nii.gz"))[0]
#         Mask_path = glob.glob(os.path.join(data_path, "MASK2.nii.gz"))[0]
#         try:
#             result = extractor.execute(T2W_HR_path, Mask_path)
#         except Exception as error:
#             print("T1 Exception occurs: ", error)
#             print("T1 error path: ", data_path)
#             errors += 1
#             continue
#         feature_size = len(result)
#         count += 1
#         # for key, val in six.iteritems(result):
#         #     pass
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Elapsed time: {elapsed_time:.6f} seconds")
#     print("feature size: ", feature_size)
#     print("T1 sample size: ", count)
#     print("T1 error size: ", errors)

# def generate_mask_data():
#     """don't need 
#     """
#     def mask(mri_path, mask_path, save_path=None):
#         # load data
#         mri_raw = nib.load(mri_path)
#         mask_raw = nib.load(mask_path)
#         mri_data = mri_raw.get_fdata()
#         mask_data = mask_raw.get_fdata()
#         # mask
#         mask_array = (mask_data > 0).astype(int)
#         masked_mri = mri_data * mask_array
#         non_zero_slices = [masked_mri[:, :, i] for i in range(masked_mri.shape[2])
#                         if np.any(masked_mri[:, :, i])]
#         result_mri = np.stack(non_zero_slices, axis=2)
#         # save
#         if save_path:
#             result_mri = nib.Nifti1Image(result_mri, mri_raw.affine, mri_raw.header)
#             nib.save(result_mri, save_path)
#             print("saved to ", save_path)
#         return result_mri

#     for data_path in tqdm(data_paths):
#         T2W_HR_path = glob.glob(os.path.join(data_path, "*T2W_HR.nii.gz"))[0]
#         Mask_path = glob.glob(os.path.join(data_path, "*MASK2.nii.gz"))[0]
#         masked = mask(T2W_HR_path, Mask_path, save_path=os.path.join(data_path, "T2W_HR_masked.nii.gz"))
#     print("Done")
