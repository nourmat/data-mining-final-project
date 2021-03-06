import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
class Visualizer():
    def __init__(self):
        matplotlib.rcParams.update({'font.size': 10})
    def boxplot(self,data,columns):
        df = pd.DataFrame(data=data,columns=columns)
        sns.boxplot(data=df)
        plt.show()
    # same as boxplot but add another dimension(box width) represnting distribution
    def violinplot(self,data,columns):
        df = pd.DataFrame(data=data,columns=columns)
        sns.violinplot(data=df)
        plt.show()
    def histogram(self,data, columns):
        df = pd.DataFrame(data,columns=columns)
        for c in columns:
            sns.distplot(df[c])
            plt.show()
    # draws a scatterplot between each pair of points and histograms as well
    def pairplot(self, data, columns):
        df = pd.DataFrame(data,columns=columns)
        sns.pairplot(df)
        plt.show()
    # draws a scatter plot between x and y to visualize the data in 2D
    def scatterplot (self,X_test,y_test):
        plt.scatter(X_test, y_test, color = 'blue')   
        #plt.plot(X_test, model.predict(X_test), color = 'red') 

