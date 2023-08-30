from sklearn import svm, preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from config import num_classes

def load_data(data_path, features):
    df = pd.read_csv(data_path)
    X = df[features].values
    y = df["label"].values
    return X, y


def process_label(y):
    """
    1. control vs. single treat vs. combination
    2. one-hot encoding label 
    """
    y[y == 2] = 1
    y[y == 3] = 2
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    return y_bin, n_classes


def train(X, y):
    # Binarize the labels for multi-class ROC
    y_bin, n_classes = process_label(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    clf = OneVsRestClassifier(svm.SVC(kernel="rbf", probability=True))
    clf.fit(X_train, y_train)
    
    # Get the predicted probabilities
    y_score = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    # # Compute the F1 score
    f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    # f1 = f1_score(y_test, y_pred, average="weighted")
    print(f'F1 score: {f1:.2f}')
    
    ## Compute the accuracy
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    # accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    # print(y_test)
    
    # roc score
    roc = roc_auc_score(y_test, y_pred, multi_class='ovr', average='micro')
    print(f'roc: {roc:.2f}')
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Plot the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr['micro'], tpr['micro'], color='blue', label='Micro-averaged ROC curve (area = %0.2f)' % roc_auc['micro'])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()


def cross_validation_roc_multi_class(
    models, X, y, n_splits=5, StratifiedK=True, smote=False, smooth=False, norm=True
):
    # y[y == 2] = 1
    # y[y == 3] = 2

    if StratifiedK:
        cv = StratifiedKFold(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits)
    # y_bin, n_classes = process_label(y) 
    mean_tpr = {}
    mean_fpr = np.linspace(0, 1, 100)
    all_tprs = {}
    aucs = {}
    accuracies = {}

    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        aucs[name] = []
        accuracies[name] = []
        all_tprs[name] = []

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train = label_binarize(y_train, classes=[0, 1, 2, 3])
            y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

            # Scaling (normalization)
            if norm:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            # Apply SMOTE to the training data (for imblanced data)
            if smote:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)
            print("y_pred", y_pred)
            print("y_score: ", y_score)
            print("y_test", y_test)
            # AUC score
            roc = roc_auc_score(y_test, y_score, multi_class='ovr', average='micro')
            print(f'{name}\'s roc: {roc:.2f}')

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            all_tprs[name].append(interp_tpr)

            # AUC score 2.0
            roc_2 = auc(fpr, tpr)
            print(f'{name}\'s roc 2.0: {roc_2:.2f}')

            aucs[name].append(auc(fpr, tpr))
            accuracies[name].append(accuracy_score(y_test, y_pred))
            print(f'{name}\'s accuracy: {accuracy_score(y_test, y_pred):.2f}')
            # accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            # print(f'Accuracy: {accuracy:.2f}')


        mean_tpr[name] = np.mean(all_tprs[name], axis=0)
        mean_tpr[name][-1] = 1.0

        if smooth:
            pass
        else:
            plt.plot(
                mean_fpr,
                mean_tpr[name],
                label=f"{name} (AUC: {np.mean(aucs[name]):.2f} ± {np.std(aucs[name]):.2f}, Accuracy: {np.mean(accuracies[name]):.2f} ± {np.std(accuracies[name]):.2f})",
            )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curves")
    plt.legend(loc="lower right")
    plt.show() 

if __name__ == "__main__":
    # features = [
    #     # "diagnostics_Mask-original_VoxelNum",
    #     "log-sigma-2-0-mm-3D_firstorder_Kurtosis",
    #     "log-sigma-2-0-mm-3D_ngtdm_Contrast.1",
    # ]
    features = ['diagnostics_Image-original_Mean', 'original_shape2D_Elongation', 
                'original_firstorder_10Percentile', 'original_firstorder_Minimum', 
                'original_ngtdm_Contrast', 'log-sigma-2-0-mm-3D_firstorder_10Percentile', 
                'log-sigma-2-0-mm-3D_firstorder_Kurtosis', 'log-sigma-3-0-mm-3D_ngtdm_Contrast', 
                'diagnostics_Image-original_Maximum.1', 'original_firstorder_Entropy.1', 
                'log-sigma-2-0-mm-3D_ngtdm_Contrast.1', 'log-sigma-4-0-mm-3D_gldm_DependenceNonUniformityNormalized.1', 
                'log-sigma-4-0-mm-3D_ngtdm_Busyness.1', 'wavelet-H_glrlm_LongRunLowGrayLevelEmphasis.1']
    multi_models ={
        "svm": OneVsRestClassifier(svm.SVC(kernel="rbf", probability=True)) 
    }
    X, y = load_data("T1_T2_outcome_processed.csv", features)
    # train(X, y)
    cross_validation_roc_multi_class(models=multi_models, X=X, y=y, smote=False, StratifiedK=True)
