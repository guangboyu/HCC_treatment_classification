import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import collections


def features_heatmap(data_path):
    
    # calculate the correlation matrix
    df = pd.read_csv(data_path)
    corr = df.corr()

    # plot the heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, xticklabels=False, yticklabels=False, cmap='YlGnBu')
    plt.title('Correlation heatmap of the features')
    plt.show()
    

def feature_count(data_path):
    """count the number of features corresponding to different type of texture feature

    Args:
        data_path (_type_): feature csv path
    """
    df = pd.read_csv(data_path)
    feature_count = collections.defaultdict(int)
    for key in df.columns:
        try:
            feature_class = key.split('_')[1]  # This assumes the key is in format "original_shape2D_Elongation"
        except:
            continue
        if feature_class in ['shape2D', 'glcm', 'firstorder', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
            feature_count[feature_class] += 1
    print(f"number of features is {len(df.columns) - 1}")
    for feature_class, count in feature_count.items():
        print(f"{feature_class}: {count} features")
        

if __name__ == '__main__':
    data_path_raw = "T2_features.csv"
    data_path = "T2_features_processed.csv"
    # features_heatmap(data_path)
    feature_count(data_path_raw)
    feature_count(data_path)