from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

LINEAR_REGRESSION = 0
POLY_REGRESSION = 1
DECISION_TREE = 2
KNN_REGRESSOR = 3

class Regressor:
    def __init__(self,type=LINEAR_REGRESSION, **kwargs):
        if type == LINEAR_REGRESSION:
            self.regressor = linear_model.LinearRegression(**kwargs)
        elif type == POLY_REGRESSION:
            self.regressor = PolynomialFeatures(**kwargs)
        elif type == DECISION_TREE:
            self.regressor = DecisionTreeRegressor(**kwargs,random_stat = 0)
        elif type == KNN_REGRESSOR:
            self.regressor = KNeighborsRegressor(**kwargs)

    def fit(self,X_train,y):
        return self.classifier.fit(X_train,y)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def score(self, X_test, y_test):
        return self.classifier.score(X_test, y_test)