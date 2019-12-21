from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

DECISION_TREE = 0
NAIVE_BAYESIAN = 1
K_NEIGHBOURS_N = 2
RANDOM_FOREST = 3

class Classifier:
    def __init__(self,type=DECISION_TREE, **kwargs):
        if type == DECISION_TREE:
            self.classifier = DecisionTreeClassifier(**kwargs)
        elif type == NAIVE_BAYESIAN:
            self.classifier = GaussianNB(**kwargs)
        elif type == K_NEIGHBOURS_N:
            self.classifier = KNeighborsClassifier(**kwargs)
        elif type == RANDOM_FOREST:
            self.classifier = RandomForestClassifier(**kwargs)

    def fit(self,X_train,y):
        return self.classifier.fit(X_train,y)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def score(self, X_test, y_test):
        return self.classifier.score(X_test, y_test)