import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns

dataset = pd.read_csv('./dataset.txt', delimiter= '\t')


def subgroup(i):
    if (i) =='HER2+':
        return 0

    elif (i) == "HR+":
        return 1

    elif (i) == "Triple Neg":
        return 2

dataset['subgroup'] = dataset['Subgroup'].apply(subgroup)


dataset.drop(['Subgroup'], axis=1, inplace=True)
dataset.to_csv('dataset_rf.csv', mode='a')

#print(dataset)
