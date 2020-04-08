import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

#read csv files
features = pd.read_csv('Train_call.txt', delimiter= '\t', index_col=0)
target = pd.read_csv('Train_clinical.txt', delimiter= '\t')
#print(features)
#print(target)
features.info()

#drop columns we wont use
for n in ['Nclone', 'Start', 'End']:
    features = features.drop(n, axis = 1)
print(features)

#transpose dataframe
features = features.transpose()
#print(features)
target = target[['Subgroup']]
#print(target)
target_np = np.array(target)
#print(target_np)
#features.info()

#add a column to the features with the target
features['Subgroup'] = target_np
dataset = features
print(dataset)

#save new dataset
dataset.to_csv(r'dataset.txt', sep='\t', mode='a')
