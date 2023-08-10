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


from config import data_paths
    
    
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


def extract_separate_T1():
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
                # create label for treat vs. untreated
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
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
    # rfecv = RFECV(estimator=model, step=1, cv=5, scoring='roc_auc', n_jobs=-1, min_features_to_select = 1)
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='roc_auc', n_jobs=-1)


    # Fit the method to the data
    rfecv = rfecv.fit(X, y)

    # Get selected features
    selected_features = list(df.columns[1:][rfecv.support_])

    print('Number of selected features: ', len(selected_features))
    print('Selected features: ', selected_features)
    
    
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
    

# def process_pipeline():
#     remove_corelation()


if __name__ == '__main__':
    T2_path = "T2_features.csv"
    T2_save_path = "T2_features_processed.csv"
    # extract_separate_T2()
    # remove_corelation(T2_path, T2_save_path)
    # remove_correlation_v2(T2_path, T2_save_path)
    feature_selection(T2_save_path)

    # extract_separate_T1()
    
    # extract_T2_all()
    # extract_T1_all()
    # check_error()