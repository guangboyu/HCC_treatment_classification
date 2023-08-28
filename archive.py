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

def extract_separate_T2(treatment_outcome=False):
    """
    To Do:
    1. Considering the influence of number of slices: only choose mask != 0 and generate more?
       2D: only one, like [160, 160, 1] -> Yes, only consider 2D now
    Params:
    treatment_outcome: treat vs untreat (False), 4 groups (True)

    """
    params = 'Params.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    start_time = time.time()
    count = 0
    errors = 0
    features = pd.DataFrame()
    for data_path in tqdm(data_paths):
        for i in range(2):
            if not glob.glob(os.path.join(data_path, f"T2W_HR_Separate_{i}.nii.gz")):
                break
            T2W_HR_path = glob.glob(os.path.join(data_path, f"T2W_HR_Separate_{i}.nii.gz"))[0]
            Mask_path = glob.glob(os.path.join(data_path, f"MASK_{i}.nii.gz"))[0]
            try:
                result = extractor.execute(T2W_HR_path, Mask_path)
                feature = pd.DataFrame([result])
                # create label for treat vs. untreated or treatment outcome
                if treatment_outcome:
                    if 'Control' in T2W_HR_path:
                        feature.insert(0, 'label', 0)
                    elif 'NK' in T2W_HR_path:
                        feature.insert(0, 'label', 1)
                    elif 'Sorafenib' in T2W_HR_path:
                        feature.insert(0, 'label', 2)
                    else:
                        feature.insert(0, 'label', 3)
                else:
                    if 'Control' in T2W_HR_path:
                        feature.insert(0, 'label', 0)
                    else:
                        feature.insert(0, 'label', 1)
                features = pd.concat([features, feature])
            except Exception as error:
                print("Exception occurs: ", error)
                print("error path: ", data_path)
                errors += 1
                continue
            feature_size = len(result)
            count += 1
            # for key, val in six.iteritems(result):
            #     pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print("feature size: ", feature_size)
    print("sample size: ", count)
    print("error size: ", errors)
    features = features.dropna(axis=1)
    # drop non-numeric columns if cannot transform to float64
    columns = features.columns
    features_numeric =  pd.DataFrame()
    for col in columns:
        try:
            df=features[col].astype(np.float64)
            features_numeric = pd.concat([features_numeric, df], axis=1)
        except:
            pass
        continue
    print("features shape: ", features_numeric.shape)
    features_numeric.to_csv("T2_features.csv", index=False)


def extract_separate_T1(treatment_outcome=False):
    """
    To Do:
    1. Considering the influence of number of slices: only choose mask != 0 and generate more?
       2D: only one, like [160, 160, 1] -> Yes, only consider 2D now
    """
    params = 'Params.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    start_time = time.time()
    count = 0
    errors = 0
    features = pd.DataFrame()
    for data_path in tqdm(data_paths):
        for i in range(2):
            if not glob.glob(os.path.join(data_path, f"T1W_HR_Separate_{i}.nii.gz")):
                break
            T1W_HR_path = glob.glob(os.path.join(data_path, f"T1W_HR_Separate_{i}.nii.gz"))[0]
            Mask_path = glob.glob(os.path.join(data_path, f"MASK_{i}.nii.gz"))[0]
            try:
                result = extractor.execute(T1W_HR_path, Mask_path)
                feature = pd.DataFrame([result])
                # create label for treat vs. untreated or treatment outcome
                if treatment_outcome:
                    if 'Control' in T1W_HR_path:
                        feature.insert(0, 'label', 0)
                    elif 'NK' in T1W_HR_path:
                        feature.insert(0, 'label', 1)
                    elif 'Sorafenib' in T1W_HR_path:
                        feature.insert(0, 'label', 2)
                    else:
                        feature.insert(0, 'label', 3)
                else:
                    if 'Control' in T1W_HR_path:
                        feature.insert(0, 'label', 0)
                    else:
                        feature.insert(0, 'label', 1)
                features = pd.concat([features, feature])
            except Exception as error:
                print("Exception occurs: ", error)
                print("error path: ", data_path)
                errors += 1
                continue
            feature_size = len(result)
            count += 1
            # for key, val in six.iteritems(result):
            #     pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print("feature size: ", feature_size)
    print("sample size: ", count)
    print("error size: ", errors)
    features = features.dropna(axis=1)
    # drop non-numeric columns if cannot transform to float64
    columns = features.columns
    features_numeric =  pd.DataFrame()
    for col in columns:
        try:
            df=features[col].astype(np.float64)
            features_numeric = pd.concat([features_numeric, df], axis=1)
        except:
            pass
        continue
    print("features shape: ", features_numeric.shape)
    features_numeric.to_csv("T1_features.csv", index=False)
    

def remove_correlation_v2(data_path, save_path):
    def correlation(dataset, threshold):
        # with the following function we can select highly correlated features
        # it will remove the first feature that is correlated with anything other feature
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
    
    df = pd.read_csv(data_path)
    col_corr = correlation(df, 0.8)
    df = df.drop(df[col_corr], axis=1)
    print(f"after remove correlation, {df.shape} features remain")
    df.to_csv(save_path, index=False)


def plot_roc_curves(metrics_dict, title):
    plt.figure(figsize=(10, 8))
    for model_name, metrics in metrics_dict.items():
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            lw=2,
            label=f'{model_name.upper()} ROC curve (area = {metrics["auc"]:.2f})',
        )
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def cross_validate(models, X, y, n_splits=5, 
                   StratifiedK=True, smote=False):
    if StratifiedK:
        cv = StratifiedKFold(n_splits=5)
    else:
        cv = KFold(n_splits=n_splits)
    # nested dict: [model_name][metric_name]
    overall_metrics = {name: {"fpr": [], "tpr": [], "auc": [], "accuracy": []} for name in models}
    avg_metrics = {name: {"fpr": [], "tpr": [], "auc": 0, "accuracy": 0} for name in models}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scaling (normalization)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        # Apply SMOTE to the training data (for imblanced data)
        if smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # trained_models = train_models(models, X_train, y_train)

        for name, model in models.items():
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val)
            overall_metrics[name]["auc"].append(metrics["auc"])
            overall_metrics[name]["fpr"].append(metrics["fpr"])
            print(f"{name} fpr:", metrics["fpr"])
            overall_metrics[name]["tpr"].append(metrics["tpr"])
            overall_metrics[name]["accuracy"].append(metrics["accuracy"])
        print("-----------")

    for name, metrics in overall_metrics.items():
        avg_metrics[name]["auc"] = sum(metrics["auc"]) / len(metrics["auc"])
        avg_metrics[name]["fpr"] = np.mean(metrics["fpr"])
        avg_metrics[name]["tpr"] = np.mean(metrics["tpr"])
        avg_metrics[name]["accuracy"] = sum(metrics["accuracy"]) / len(metrics["accuracy"])
        avg_auc = avg_metrics[name]["auc"]
        avg_acc = avg_metrics[name]["accuracy"]
        print(f"Average AUC for {name.upper()}: {avg_auc:.4f}")
        print(f"Average Accuracy for {name.upper()}: {avg_acc:.4f}")
        print("-----------------------")

    return overall_metrics