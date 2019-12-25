from sklearn.cluster import KMeans
from sklearn import datasets

KMEANS = 0

class Clustering:
    def __init__(self,type=KMEANS, **kwargs):
        if type == KMEANS:
            self.cluster = KMeans(**kwargs)
        
    def fit(self,X_train):
        return self.cluster.fit(X_train)

    def predict(self, X_test):
        return self.cluster.predict(X_test)

    def score(self, X_test):
        return self.cluster.score(X_test)
    

#preprosseing    
iris = datasets.load_iris()
x = iris.data
y = iris.target

params = dict(n_clusters=3)

model = Clustering(type = 0, **params)
model.fit(x)

print(y)
print (model.cluster.labels_)