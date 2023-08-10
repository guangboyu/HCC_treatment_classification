import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def modeling():
    # Load data
    df = pd.read_csv('T2_features_processed.csv')
    
    # specify your selected features
    # selected_features = ['diagnostics_Mask-original_VoxelNum', 'original_shape2D_Elongation', 
    #                      'log-sigma-3-0-mm-3D_firstorder_Kurtosis', 'log-sigma-5-0-mm-3D_glrlm_LongRunHighGrayLevelEmphasis', 
    #                      'wavelet-H_firstorder_Kurtosis']
    # selected_features = ['diagnostics_Mask-original_VoxelNum', 'original_shape2D_Elongation', 'log-sigma-3-0-mm-3D_firstorder_Kurtosis', 'log-sigma-3-0-mm-3D_glrlm_RunLengthNonUniformity', 'log-sigma-5-0-mm-3D_glcm_Imc1', 'log-sigma-5-0-mm-3D_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-H_firstorder_Kurtosis', 'wavelet-H_glcm_ClusterShade']
    selected_features = ['diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_VoxelNum', 'original_shape2D_Elongation', 
                         'log-sigma-2-0-mm-3D_ngtdm_Contrast', 'log-sigma-5-0-mm-3D_glrlm_LongRunHighGrayLevelEmphasis']
    # Separate the features from the target and focus only on selected features
    X = df[selected_features].values
    y = df['label'].values  # assuming the target is in 'label' column

    # # Split the dataframe into features and target
    # X = df.drop('label', axis=1) # assuming 'label' is the column with your target values
    # y = df['label']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Length of X train {len(X_train)}")
    print(f"Length of X train {len(X_val)}")

    # Scale the features
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    
    # # Apply SMOTE to the training data (for imblanced data)
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # # Apply PCA
    # pca = PCA(n_components=5)
    # X_train = pca.fit_transform(X_train)
    # X_val = pca.transform(X_val)

    # Train the model with svm
    svm_clf = svm.SVC(kernel='rbf', probability=True)
    svm_clf.fit(X_train, y_train)
    print("Accuracy of SVM model is: ")
    svm_metric_train = evaluate(svm_clf, X_train, y_train)
    svm_metric_val = evaluate(svm_clf, X_val, y_val)

    # plot_weight(svm_clf, selected_features)


    # Train the model with xgboost
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
    xgb_clf.fit(X_train, y_train)
    print("Accuracy of XGBoost model is: ")
    xgb_metric_train = evaluate(xgb_clf, X_train, y_train)
    xgb_metric_val = evaluate(xgb_clf, X_val, y_val)
    
    # Train the model with RF
    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train, y_train)
    print("Accuracy of RF model is: ")
    rf_metric_train = evaluate(rf_clf, X_train, y_train)
    rf_metric_val = evaluate(rf_clf, X_val, y_val)
    
    # Plot the ROC curves
    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.figure(figsize=(10, 8))
    plt.plot(xgb_metric_train["fpr"], xgb_metric_train["tpr"], color='blue', lw=2, label=f'XGBoost ROC curve (area = {xgb_metric_train["auc"]:.2f})')
    plt.plot(svm_metric_train["fpr"], svm_metric_train["tpr"], color='green', lw=2, label=f'SVM ROC curve (area = {svm_metric_train["auc"]:.2f})')
    plt.plot(rf_metric_train["fpr"], rf_metric_train["tpr"], color='yellow', lw=2, label=f'RF ROC curve (area = {rf_metric_train["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic: Training')
    plt.legend(loc='lower right')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    plt.plot(xgb_metric_val["fpr"], xgb_metric_val["tpr"], color='blue', lw=2, label=f'XGBoost ROC curve (area = {xgb_metric_val["auc"]:.2f})')
    plt.plot(svm_metric_val["fpr"], svm_metric_val["tpr"], color='green', lw=2, label=f'SVM ROC curve (area = {svm_metric_val["auc"]:.2f})')
    plt.plot(rf_metric_val["fpr"], rf_metric_val["tpr"], color='yellow', lw=2, label=f'RF ROC curve (area = {rf_metric_val["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic: Validation')
    plt.legend(loc='lower right')
    plt.show()
    

def evaluate(model, X_val, y_val):
    # Make predictions on the validation set
    y_val_pred = model.predict_proba(X_val)[:,1]
    y_val_class = model.predict(X_val)
    # get accuracy
    accuracy = accuracy_score(y_val, y_val_class)
    print(f"{accuracy}")
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
    roc_auc = auc(fpr, tpr)
    matrics = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "accuracy": accuracy
    }
    return matrics

def plot_weight(clf, feature_names):
    # Get the weights (coefficients) of the features
    feature_weights = -clf.coef_[0]
    feature_names = [x.split("_", 1)[-1] for x in feature_names]
    # Plot the weights
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_weights)
    plt.xticks(rotation=90) # Rotate the feature names for readability
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.title('Feature Weights')
    plt.show()

    
    
if __name__ == '__main__':
    modeling()