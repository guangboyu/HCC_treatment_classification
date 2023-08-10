import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import collections
import numpy as np


def features_heatmap(data_path):
    
    # calculate the correlation matrix
    df = pd.read_csv(data_path)
    corr = df.corr()

    # plot the heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, 
                vmin=-1, vmax=1, center=0,
                annot=False, xticklabels=False, yticklabels=False, 
                cmap=sns.diverging_palette(20, 220, n=200), square=True)
    plt.title('Correlation heatmap of the features')
    plt.show()


def features_heatmap_3D(data_path):
    df = pd.read_csv(data_path)
    corr_matrix = df.corr()
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xlabels = np.arange(corr_matrix.shape[0])
    ylabels = np.arange(corr_matrix.shape[1])
    xpos, ypos = np.meshgrid(xlabels, ylabels, copy=False, indexing="ij")

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = 0.75
    dy = 0.75
    dz = corr_matrix.values.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=plt.cm.viridis(dz))

    # Set the ticks and labels
    ax.w_xaxis.set_ticks(xlabels + 0.5 / 2.)
    ax.w_yaxis.set_ticks(ylabels + 0.5 / 2.)
    ax.w_xaxis.set_ticklabels(corr_matrix.columns)
    ax.w_yaxis.set_ticklabels(corr_matrix.columns)

    # Labeling and title
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Correlation')
    ax.set_title('3D Correlation Heatmap')
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
    features_heatmap(data_path_raw)
    # features_heatmap_3D(data_path)
    # feature_count(data_path_raw)
    # feature_count(data_path)