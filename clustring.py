from sklearn.cluster import KMeans

KMEANS = 0

class Clustering:
    def __init__(self,type=KMEANS, **kwargs):
        if type == KMEANS:
            self.clustering= KMeans(**kwargs)
        
    def fit(self,X_train,y):
        return self.classifier.fit(X_train,y)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def score(self, X_test, y_test):
        return self.classifier.score(X_test, y_test)