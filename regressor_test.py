import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from preprocessing import Preprocessing
import regressor as reg
import visualizer as vis
#preprocssing
dataset = pd.read_csv('diamonds.csv')

prepro = Preprocessing()
prepro.handleMissing(dataset)

x = dataset.drop(['price','cut','color','clarity'],axis = 1)
y = dataset['price']

x = prepro.scale(x)

encode_col = dataset[['cut','color','clarity']]
encode_col  = prepro.encode(encode_col)

x = np.concatenate((x,encode_col),axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=0,test_size=0.33)

vis.Visualizer().scatterplot(X_test[:,0],y_test.iloc[:])

# Linear Regression
regressor = reg.Regressor(type=reg.LINEAR_REGRESSION)
regressor.fit(X_train, y_train)
print("******************Linear Regression******************")
print(regressor.score(X_test,y_test))
#vis.Visualizer().scatterplot(X_test[:,0],y_test.iloc[:],regressor)
print("*************************************************")



# polynomial Regression
params = dict(degree = 5)
regressor = reg.Regressor(type=reg.POLY_REGRESSION, **params)
regressor.fit(X_train, y_train)
print("**************Polynomial Regression***************")
#print(regressor.score(X_test,y_test))
print("*************************************************")


# Decision Tree
params = dict(criterion='mse')
regressor = reg.Regressor(type=reg.DECISION_TREE, **params)
regressor.fit(X_train, y_train)
print("******************Decision Tree******************")
print(regressor.score(X_test,y_test))
print("*************************************************")


# KNN REgressor
# params = dict()
params = dict(n_neighbors = 5)
regressor = reg.Regressor(type=reg.KNN_REGRESSOR, **params)
regressor.fit(X_train, y_train)
print("****************KNN Regressor*******************")
print(regressor.score(X_test,y_test))
print("*************************************************")
