import preprocessing as pp
import classifier as cc
import visualizer as vs
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


cols = [i for i in range(0,32)]
df = pd.read_csv('breast_cancer.csv',index_col=32).drop("id",axis=1).drop("diagnosis",axis=1)
cols = df.columns
for i in df.columns:
    if "mean" not in i:
        df = df.drop(i,axis=1)
vs.Visualizer().boxplot(df.values,df.columns)
vs.Visualizer().violinplot(df.values,df.columns)
vs.Visualizer().pairplot(df.values,df.columns)
vs.Visualizer().histogram(df.values,df.columns)