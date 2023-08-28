import pandas as pd
import numpy as np

from sklearn import svm, preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import xgboost as xgb

from config import num_classes

def load_data(data_path, features):
    df = pd.read_csv(data_path)
    X = df[features].values
    y = df["label"].values
    return X, y


def evaluate(model, X, y):
    y_pred = model.predict_proba(X)[:, 1]
    y_class = model.predict(X)
    accuracy = accuracy_score(y, y_class)
    # print(f"{accuracy}")
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "accuracy": accuracy}


def cross_validation_roc(
    models, X, y, n_splits=5, StratifiedK=True, smote=False, smooth=False, norm=True
):
    if StratifiedK:
        cv = StratifiedKFold(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits)

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
            y_score = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_score)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            all_tprs[name].append(interp_tpr)

            aucs[name].append(auc(fpr, tpr))
            accuracies[name].append(accuracy_score(y_test, y_pred))

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
    features = [
        # "diagnostics_Mask-original_VoxelNum",
        "log-sigma-2-0-mm-3D_firstorder_Kurtosis",
        "log-sigma-2-0-mm-3D_ngtdm_Contrast.1",
    ]
    X, y = load_data("T1_T2_treatment_processed.csv", features)

    models = {
        "svm": svm.SVC(kernel="rbf", probability=True),
        "xgboost": xgb.XGBClassifier(objective="binary:logistic", n_estimators=100),
        "rf": RandomForestClassifier(n_estimators=100),
        "glm": LogisticRegression(),
        "gbd": GradientBoostingClassifier(),
        "NN": MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1
        ),
    }
    
    multi_class_models = {
        "svm": OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True))
    }

    cross_validation_roc(models, X, y, StratifiedK=True, smote=True, norm=False)
