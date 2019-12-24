import preprocessing as pp
import classifier as cc
import visualizer as vs
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

cols = [i for i in range(0,32)]
df = pd.read_csv('breast_cancer.csv',index_col=32)
df = df.drop('id',axis=1)
df = pp.Preprocessing().handleMissing(df)
X = df.drop('diagnosis',axis=1)
X = pp.Preprocessing().scale(X.values, type=pp.STANDARD_SCALER)
y = df['diagnosis']
y = pp.Preprocessing().encode(y.values, type=pp.LABEL_ENCODER)

X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis',axis=1), y,random_state=0,test_size=0.33)

# Random Forest
params = dict(n_estimators=95, random_state=0, criterion='gini')
classifier = cc.Classifier(type=cc.RANDOM_FOREST, **params)
classifier.fit(X_train, y_train)
print("******************Random Forest******************")
print(classifier.score(X_test,y_test))
print("*************************************************")


# Decision Tree
params = dict(random_state=0, criterion='gini')
classifier = cc.Classifier(type=cc.DECISION_TREE, **params)
classifier.fit(X_train, y_train)
print("******************Decision Tree******************")
print(classifier.score(X_test,y_test))
print("*************************************************")


# KNN
params = dict(n_neighbors = 34)
classifier = cc.Classifier(type=cc.K_NEIGHBOURS_N, **params)
classifier.fit(X_train, y_train)
print("*********************KNN*************************")
print(classifier.score(X_test,y_test))
print("*************************************************")




# Naive Bayesian
# params = dict()
params = dict(var_smoothing=0.00002)
classifier = cc.Classifier(type=cc.NAIVE_BAYESIAN, **params)
classifier.fit(X_train, y_train)
print("****************Naive Bayesian*******************")
print(classifier.score(X_test,y_test))
print("*************************************************")





