from sklearn import preprocessing as skp
import numpy as np
import pandas as pd
STANDARD_SCALER = 0
MIN_MAX_SCALER = 1
MIN_SCALER = 2
MAX_SCALER = 3

ORDINAL_ENCODER = 4
LABEL_ENCODER = 5
ONE_HOT_ENCODER = 6

DROP_MISSING = 7
FILL_MEAN = 8
FILL_MODE = 9
class Preprocessing:
    def scale(self,X_train:np.array, type=STANDARD_SCALER):
        ans = None
        if type == STANDARD_SCALER:
            ans = skp.StandardScaler().fit_transform(X_train)
        elif type == MIN_MAX_SCALER:
            ans = skp.MinMaxScaler().fit_transform(X_train)
        elif type == MIN_SCALER:
            ans = X_train.copy()
            for j in range(ans.shape[1]):
                min = ans.T[j].min()
                for i in range(ans.shape[0]):
                    if ans[i][j] == 0: ans[i][j]=np.inf
                    else: ans[i][j] = min / ans[i][j]
        elif type == MAX_SCALER:
            ans = X_train.copy()
            for j in range(ans.shape[1]):
                max = ans.T[j].max()
                for i in range(ans.shape[0]):
                    ans[i][j] = ans[i][j] / max
        return ans
    def encode(self, X_train, type=ORDINAL_ENCODER):
        ans = None
        if type == ORDINAL_ENCODER:
            ans = skp.OrdinalEncoder().fit_transform(X_train)
        elif type == LABEL_ENCODER:
            ans = skp.LabelEncoder().fit_transform(X_train)
        elif type == ONE_HOT_ENCODER:
            ans = skp.OneHotEncoder().fit_transform(X_train)
        return ans
    def handleMissing(self, df, type=DROP_MISSING):
        ans = None
        if(type == DROP_MISSING):
            ans = df.dropna()
        elif(type == FILL_MEAN):
            ans = df.fillna(df.mean())
        elif(type == FILL_MODE):
            ans = df.fillna(df.mode())
        return ans