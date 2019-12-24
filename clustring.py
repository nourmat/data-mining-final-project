from sklearn.cluster import KMeans

KMEANS = 0

class Clustering:
    def __init__(self,type=KMEANS, **kwargs):
        if type == KMEANS:
            self.cluster = KMeans(**kwargs)
        
    def fit(self,X_train,y):
        return self.cluster.fit(X_train,y)

    def predict(self, X_test):
        return self.cluster.predict(X_test)

    def score(self, X_test, y_test):
        return self.cluster.score(X_test, y_test)