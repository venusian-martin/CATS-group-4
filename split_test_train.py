import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

dataset = pd.read_csv('./dataset.txt', delimiter= '\t', index_col = 0)
print(dataset)

def subgroup(i):
    if (i) =='HER2+':
        return 0

    elif (i) == "HR+":
        return 1

    elif (i) == "Triple Neg":
        return 2

dataset['subgroup'] = dataset['Subgroup'].apply(subgroup)

dataset.drop(['Subgroup'], axis=1, inplace=True)
print(dataset)
#dataset.to_csv('dataset_rf.csv', mode='a')


#Are the classes balanced?
subgroup = dataset['subgroup']
print(subgroup)
features = dataset.drop(['subgroup'], axis = 1)
print(features)
sub_id, freq = np.unique(subgroup, return_counts=True) #count frequency of each class
#plotting frequency of each class in a pie chart
explode = (0.1, 0.1, 0.1) #separate one of the pie pieces
labels = ['HER2+', 'HR+', 'Triple Neg']
fig1, ax1 = plt.subplots()
ax1.pie(freq, explode=explode, labels=labels, startangle=90, autopct='%1.1f%%', shadow =True)
ax1.axis('equal')
plt.show()

#standardization
#subgroup = preprocessing.scale(subgroup)
#print(target)
#features = preprocessing.scale(features)
#print(features)

#normalization
#subgroup = preprocessing.normalize(subgroup)
#print(target)
#features = preprocessing.normalize(features)
#print(features)

from sklearn.model_selection import train_test_split #Split data
x_train, x_test, y_train_classification, y_test_classification = train_test_split(features, subgroup, test_size=0.2) #Split in training and test set

y_train = np.array(y_train_classification)
y_test = np.array(y_test_classification)


#building datasets and saving them
Test_Classification = pd.concat([x_test, y_test_classification], axis=1, join='inner')
print(Test_Classification)
Test_Classification.to_csv(r'Test_Classification.csv',index=False)

Training_Classification = pd.concat([x_train, y_train_classification], axis=1, join='inner')
print(Training_Classification)
Training_Classification.to_csv(r'Training_Classification.csv',index=False)
