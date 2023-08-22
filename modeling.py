import pandas as pd

from sklearn import svm, preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import xgboost as xgb


def load_data(data_path, features):
    df = pd.read_csv(data_path)
    X = df[features].values
    y = df["label"].values
    return X, y


def train_models(X_train, y_train):
    models = {
        "svm": svm.SVC(kernel="rbf", probability=True),
        "xgboost": xgb.XGBClassifier(objective="binary:logistic", n_estimators=100),
        "rf": RandomForestClassifier(n_estimators=100),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Accuracy of {name.upper()} model is:")
        evaluate(model, X_train, y_train)

    return models


def evaluate(model, X, y):
    y_pred = model.predict_proba(X)[:, 1]
    y_class = model.predict(X)
    accuracy = accuracy_score(y, y_class)
    print(f"{accuracy}")
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "accuracy": accuracy}


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
    overall_metrics = {name: [] for name in models}

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

        trained_models = train_models(X_train, y_train)

        for name, model in trained_models.items():
            metrics = evaluate(model, X_val, y_val)
            overall_metrics[name].append(metrics["auc"])

    for name, metrics in overall_metrics.items():
        avg_auc = sum(metrics) / len(metrics)
        print(f"Average AUC for {name.upper()}: {avg_auc:.4f}")

    return overall_metrics


if __name__ == "__main__":
    features = [
        # "diagnostics_Mask-original_VoxelNum",
        "log-sigma-2-0-mm-3D_firstorder_Kurtosis",
        "log-sigma-2-0-mm-3D_glcm_Contrast",
    ]
    X, y = load_data("T1_T2_treatment_processed.csv", features)

    models = {
        "svm": svm.SVC(kernel="rbf", probability=True),
        "xgboost": xgb.XGBClassifier(objective="binary:logistic", n_estimators=100),
        "rf": RandomForestClassifier(n_estimators=100),
    }

    cross_validate(models, X, y, 
                   StratifiedK=True, smote=True)
