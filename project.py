import preprocessing as pp
import classifier as cc
import visualizer as vs
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

X = datasets.load_iris()
# df = pd.DataFrame(data=X.data,columns=X.feature_names)
V = vs.Visualizer()
V.heatmap(X.data, X.feature_names)