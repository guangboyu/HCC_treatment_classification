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
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb


from config import data_paths, treatment_path
    
    
def extract_features(modality="T2", treatment_outcome=False, data_paths=[]):
    """
    Extracts radiomic features from MRI data.

    Parameters:
        - modality: 'T1' or 'T2'
        - treatment_outcome: If False, assigns labels based on treated vs untreated. 
                             If True, assigns labels based on treatment type.
        - data_paths: List of paths containing the MRI data
    """

    params = 'Params.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    start_time = time.time()
    count = 0
    errors = 0
    features_df = pd.DataFrame()

    for data_path in tqdm(data_paths):
        for i in range(2):
            image_path = glob.glob(os.path.join(data_path, f"{modality}W_HR_Separate_{i}.nii.gz"))
            mask_path = glob.glob(os.path.join(data_path, f"MASK_{i}.nii.gz"))

            if not image_path:
                break

            try:
                result = extractor.execute(image_path[0], mask_path[0])
                feature = pd.DataFrame([result])

                # Assign labels
                if treatment_outcome:
                    treatment_labels = {'Control': 0, 'NK': 1, 'Sorafenib': 2, 'Combination': 3}
                    label = treatment_labels.get(next(filter(lambda x: x in image_path[0], treatment_labels)), 3)
                else:
                    label = 0 if 'Control' in image_path[0] else 1

                feature.insert(0, 'label', label)
                features_df = pd.concat([features_df, feature])

            except Exception as error:
                print("Exception occurs: ", error)
                print("Error path: ", data_path)
                errors += 1
                continue

            count += 1

    elapsed_time = time.time() - start_time

    print_stats(elapsed_time, len(result), count, errors)

    filename = ""
    if treatment_outcome:
        filename = f"{modality}_outcome_features.csv"
        save_features(features_df, filename)
    else:
        filename = f"{modality}_treatment_features.csv"
        save_features(features_df, filename)
    return filename


def print_stats(elapsed_time, feature_size, count, errors):
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print("Feature size: ", feature_size)
    print("Sample size: ", count)
    print("Error size: ", errors)


def save_features(features, filename):
    features.dropna(axis=1, inplace=True)
    features_numeric = features.apply(pd.to_numeric, errors='coerce')
    features_numeric.dropna(axis=1, inplace=True)
    print("Features shape: ", features_numeric.shape)
    features_numeric.to_csv(filename, index=False)



def remove_corelation(data_path, save_path):
    """
    remove features with high correlation
    r = 0.8
    """
    df = pd.read_csv(data_path)
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than a certain threshold (e.g., 0.95)
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

    # Drop highly correlated features 
    df = df.drop(df[to_drop], axis=1)
    
    print(f"after remove correlation, {df.shape} features remain")

    # Save the processed dataframe to a new csv file
    df.to_csv(save_path, index=False)


def feature_selection(data_path, visualize=False):
    """
    last step of feature selection
    """
    # Load the data
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop('label', axis=1) 
    y = df['label']

    # Scaling the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define the method
    # model = LogisticRegression(max_iter=10000)
    # model = svm.SVC(kernel='linear', probability=True)
    if "outcome" in data_path:
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, n_estimators=100)
    else:
        model = xgb.XGBClassifier(n_estimators=100)
    # rfecv = RFECV(estimator=model, step=1, cv=5, scoring='roc_auc', n_jobs=-1, min_features_to_select = 1)
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='roc_auc', n_jobs=-1)


    # Fit the method to the data
    rfecv = rfecv.fit(X, y)

    # Get selected features
    selected_features = list(df.columns[1:][rfecv.support_])
    print("Data Path: ", data_path)
    print('Number of selected features: ', len(selected_features))
    print('Selected features: ', selected_features)
    print("----------------------------------------------------")
    
    
    # assuming df is your DataFrame and it's already loaded

    # create a smaller dataframe with just the selected features
    df_selected = df.drop('label', axis=1)  # assuming the target column is 'label'

    # calculate the correlation matrix
    corr = df_selected.corr()

    # # create a mask to hide the upper triangle of the correlation matrix (since it's mirrored around its main diagonal)
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    # # generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # # plot the heatmap
    # plt.figure(figsize=(12,10))
    # sns.clustermap(corr, mask=mask, cmap=cmap, center=0, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.title('Clustered correlation heatmap of the features')
    # plt.show()
            


def check_error():
    # Bounding ROI
    mri_path = glob.glob("Data/Combination/B35R2_101922/*T2W_HR.nii.gz")[0]
    mask_path = ("Data/Combination/B35R2_101922/MASK2.nii.gz")
    # check dimension match
    mri_data = nib.load(mri_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    print("MRI dimension: ", mri_data.shape)
    print("mask dimension: ", mask_data.shape)


def merge_features(data_path_1, data_path_2, save_path):
    # Read the two feature files
    t1_features = pd.read_csv(data_path_1)
    t2_features = pd.read_csv(data_path_2)

    # Ensure the rows of the two dataframes match up. 
    # This assumes that the rows are in the same order and the dataframes have the same length.
    # If they have an ID or some unique identifier, you should use that for merging instead.

    if len(t1_features) != len(t2_features):
        raise ValueError("The two CSVs have a different number of rows!")

    # Merge the two dataframes horizontally
    # This will create a dataframe with columns from both original dataframes side by side
    merged_features = pd.concat([t1_features, t2_features], axis=1)

    # Optionally, if you have some column like 'ID' which repeats in both files, you can drop one of them
    # merged_features = merged_features.loc[:,~merged_features.columns.duplicated()]

    # Save the merged features to a new CSV
    merged_features.to_csv(save_path, index=False)



# def process_pipeline():
#     remove_corelation()


if __name__ == '__main__':
    # # generate raw features
    # T1_outcome = extract_features(modality="T1", data_paths=data_paths, treatment_outcome=True)
    # T1_treatment = extract_features(modality="T1", data_paths=data_paths, treatment_outcome=False)
    # T2_outcome = extract_features(modality="T2", data_paths=data_paths, treatment_outcome=True)
    # T2_treatment = extract_features(modality="T2", data_paths=data_paths, treatment_outcome=False)
    # T1_T2_outcome = merge_features(T1_outcome, T2_outcome, "T1_T2_outcome.csv")
    # T1_T2_treatment = merge_features(T1_treatment, T2_treatment, "T1_T2_treatment.csv")
    # # remove correlation
    # for path in glob.glob("*.csv"):
    #     remove_corelation(path, path.replace(".csv", "_processed.csv"))
    # feature selection
    for path in treatment_path:
        feature_selection(path)