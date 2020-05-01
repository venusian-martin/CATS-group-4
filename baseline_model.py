import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy.random import seed, randn
from sklearn.metrics import accuracy_score

#load dataset
dataset = pd.read_csv('./dataset.txt', delimiter= '\t', index_col = 0)
print(dataset)

#dividing dataset into 3 'equal' parts
train_1 = dataset.index[0:33]
train_2 = dataset.index[33:66]
train_3 = dataset.index[66:100]

test_1 = dataset.iloc[0:33]
test_2 = dataset.iloc[33:66]
test_3 = dataset.iloc[66:100]

train = [train_1, train_2, train_3]
test = [test_1, test_2, test_3]

######## BASELINE MODEL #############
#defining list to append accuracy values per splitted data
ACCURACY_SUMMARY = []

#defining dataframe to save predictions for each splitted data
Predictions_total = pd.DataFrame()

############################### 3 FOLD CROSS VALIDATION #########################################
for n, m in zip(train,test):

    #### TRAINING #####
    training_fold = dataset.drop(n) #define training set
    print(training_fold)

    #Selecting our features and our target for each training set
    y_train = np.ravel(training_fold[['Subgroup']]) #target
    x_train = training_fold.drop(['Subgroup'],axis=1) #features

    test_fold = m #define test set

    #Selecting our features and our target for each test set
    y_test = np.ravel(test_fold[['Subgroup']]) #target
    x_test = test_fold.drop(['Subgroup'],axis=1) #features

    # zero rule algorithm for classification: predicts same subtype for every sample
    def zero_rule_algorithm_regression(train, test, subtype):
        predicted = [subtype for i in range(len(test_fold))]
        return predicted

    subtype = 'HER2+'
    y_pred = zero_rule_algorithm_regression(training_fold, test_fold, subtype)

    #compute accuracy for fold
    ACC_fold = accuracy_score(np.array(y_test), np.array(y_pred))
    ACCURACY_SUMMARY.append(ACC_fold) #append accuracies for each repetion

    #dataframe with predictions vs actual for each repetition
    predictions_fold = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred})

    #save predictions for each fold
    Predictions_total = pd.concat([Predictions_total, predictions_fold], axis = 0)

print(Predictions_total)
print(ACCURACY_SUMMARY)

ACCURACY_TABLE = pd.DataFrame({'Fold': [1,2,3], 'Accuracy': ACCURACY_SUMMARY})

#final mean accuracy (3 folds)
ACCURACY_FINAL = np.mean(ACCURACY_SUMMARY)
print('Mean final accuracy:')
print(ACCURACY_FINAL)

#save a files with the predictions and accuracies
ACCURACY_TABLE.to_csv(r'accuracies_baseline_HER2.txt', sep='\t', mode = 'a')
Predictions_total.to_csv(r'Predictions_baseline_HER2.txt', sep='\t', mode='a')
