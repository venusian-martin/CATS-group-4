from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy.random import seed, randn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, GridSearchCV
import seaborn as sns
from numpy.random import seed
from sklearn.metrics import confusion_matrix

#read dataset
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
#folds = [0,1,2]

#number repitions array
#rep = np.array([1,2,3,4,5,6,7,8,9,10])
rep = np.array(range(1,51))

#initialize dataframe to save best hyperparameters
Best_hyperparam = pd.DataFrame()

#defining list to append accuracy values per splitted data
ACCURACY_SUMMARY = []

#defining dataframe to save predictions and feature importance for each splitted data
Predictions_total = pd.DataFrame()
Feature_importance_total = pd.DataFrame()


############################### 3 FOLD OUTER LOOP ##########################################
for n, m in zip(train,test):

    #### TRAINING #####
    training_fold = dataset.drop(n) #define training set
    print(training_fold)

    #Selecting our features and our target for each training set
    y_train = np.ravel(training_fold[['Subgroup']]) #target
    x_train = training_fold.drop(['Subgroup'],axis=1) #features

    #initialize hyperparameters
    n_estimators_fold = []
    min_sample_split_fold = []
    max_features_fold = []

    # initialize dataframe to save predictions for each FOLD
    Predictions_fold = pd.DataFrame()

    # initialize dataframe to save feature importance for each FOLD
    Feature_importance_fold = pd.DataFrame()

    # initialize list to save accuracies for each FOLD
    Accuracy_score_fold = []

    seed(500) #prevent rf randomness

    ################## REPETITIONS LOOP WITHIN EACH FOLD ##########################
    for loop in range(0,50):

        ### Random Forest hyperparameter tuning through grid search with K fold cross validation (k=5) ###
        parameters={'min_samples_split' : range(2,12,2), 'n_estimators': range(100,130,10), 'max_features': range(50,90,10)}
        clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring = 'accuracy')
        clf.fit(x_train, y_train) #fitting the model
        print("Best parameters set found on development set:")
        print(clf.best_params_,clf.best_score_)

        #saving the best hyperparameters
        min_sample_split_rep = clf.best_params_['min_samples_split']
        n_estimators_rep = clf.best_params_['n_estimators']
        max_features_rep = clf.best_params_['max_features']

        #saving best hyperparameters
        n_estimators_fold.append(n_estimators_rep)
        min_sample_split_fold.append(min_sample_split_rep )
        max_features_fold.append(max_features_rep)

######################## VALIDATION/TEST SET ##################################

        test_fold = m #define test set

        #Selecting our features and our target for each training set
        y_test = np.ravel(test_fold[['Subgroup']]) #target
        x_test = test_fold.drop(['Subgroup'],axis=1) #features


        #Random Forest with tuned hyperparameters for each repetition
        parameters={'min_samples_split': [min_sample_split_rep] ,'max_features': [max_features_rep] , 'n_estimators': [n_estimators_rep]}
        clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring = 'accuracy')
        clf.fit(x_train, y_train) #fitting the model

        #Plot features importance
        features = (clf.best_estimator_.feature_importances_)
        print(features)
        chromosomesid, importance, chromosomes = [], [], []
        for f in range(len(features)):
            if features[f] >= 0.005:
                importance.append(features[f])
                chromosomesid.append(x_train.columns[f])
                chromosomes.append(x_train.columns[f])

        #feature importance of each repetition
        Feature_importance_rep = pd.DataFrame({'Chromosome id': chromosomesid, 'Importance': importance})
        #saving it into fold feature importance dataframe
        Feature_importance_fold = pd.concat([Feature_importance_fold, Feature_importance_rep], axis = 1)


        ##### Predicting #####
        y_pred = clf.predict(x_test) # predict from test features

        #dataframe with predictions vs actual for each repetition
        predictions_rep = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
        #compute accuracy of the repetition
        ACC_rep = accuracy_score(np.array(y_test), np.array(y_pred))
        Accuracy_score_fold.append(ACC_rep) #append accuracies for each repetion
        Predictions_fold = pd.concat([Predictions_fold, predictions_rep], axis = 1) #append predictions for each repetition

    #save best hyperparameters per rep for each fold into dataframe
    Best_hyperparam_fold = pd.DataFrame({'Repetition': rep,
                                        'min_samples_split': min_sample_split_fold,
                                        'max_features': max_features_fold,
                                        'n_estimators': n_estimators_fold})

    #Dataframe with all best hyperparameters for each fold
    Best_hyperparam = pd.concat([Best_hyperparam, Best_hyperparam_fold], axis = 0)
    #save predictions for each fold
    Predictions_total = pd.concat([Predictions_total, Predictions_fold], axis = 0)
    #save feature importance for each fold
    Feature_importance_total = pd.concat([Feature_importance_total, Feature_importance_fold], axis = 0)
    #dataframe with accuracy per repetition
    ACCURACY_FOLD = pd.DataFrame({'Repetition': rep, 'Accuracy': Accuracy_score_fold})
    #compute mean accuracy per fold
    mean_acc_fold = np.mean(Accuracy_score_fold)
    #save mean accuracy per fold
    ACCURACY_SUMMARY.append(mean_acc_fold)

print(Best_hyperparam)
print(Predictions_total)
print(Feature_importance_total)
print(ACCURACY_SUMMARY)

#final mean accuracy (3 folds)
ACCURACY_FINAL = np.mean(ACCURACY_SUMMARY)
print('Mean final accuracy:')
print(ACCURACY_FINAL)

#save a file with the best hyperparameters, the feature importance and with the predictions
Best_hyperparam.to_csv(r'best_hyperparameters.txt', sep='\t', mode = 'a')
Feature_importance_total.to_csv(r'feature_importance.txt', sep='\t', mode='a')
Predictions_total.to_csv(r'Predictions_total.txt', sep='\t', mode='a')
