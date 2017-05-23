import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn

import seaborn as sns

# load dataset from sklearn
from sklearn.datasets import load_boston
boston = load_boston()

def normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu)/sigma

if __name__ == '__main__':

    # load features and labels
    features = np.array(boston.data)
    labels = np.array(boston.target)

    # normalize features
    features = normalize(features)

    # split into training and testing (80/20 split)
    features_train = features[:int(len(features)*0.8)]
    features_test = features[len(features_train):]

    labels_train = labels[:int(len(labels)*0.8)]
    labels_test = labels[len(labels_train):]

# leave off for now


