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
            min = X_train.min()
            ans = np.array([min / xi for xi in X_train])
        elif type == MAX_SCALER:
            max = X_train.max()
            ans = np.array([max / xi for xi in X_train])
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
    def handleMissing(self, X_train, type=DROP_MISSING):
        df = pd.DataFrame(X_train)
        ans = None
        if(type == DROP_MISSING):
            df.dropna().to_numpy()
        elif(type == FILL_MEAN):
            ans = df.fillna(df.mean()).to_numpy()
        elif(type == FILL_MODE):
            ans = df.fillna(df.mode()).to_numpy()
        return ans