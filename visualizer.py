import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
class Visualizer():
    def __init__(self):
        matplotlib.rcParams.update({'font.size': 10})
    def boxplot(self,data,columns):
        plt.xticks(rotation=30)
        df = pd.DataFrame(data=data,columns=columns)
        sns.boxplot(data=df)
        plt.show()
        plt.xticks(rotation=0)
    # same as boxplot but add another dimension(box width) represnting distribution
    def violinplot(self,data,columns):
        plt.xticks(rotation=30)
        df = pd.DataFrame(data=data,columns=columns)
        sns.violinplot(data=df)
        plt.show()
        plt.xticks(rotation=0)
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
