import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
class Visualizer():
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
