from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy.random import seed, randn
from sklearn.metrics import roc_auc_score, make_scorer, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, GridSearchCV
import seaborn as sns
from numpy.random import seed

train = ['Train1.txt', 'Train2.txt', 'Train3.txt', 'Train4.txt', 'Train5.txt',
            'Train6.txt', 'Train7.txt', 'Train8.txt','Train9.txt','Train10.txt'] #our training sets

set = np.array([1,2,3,4,5,6,7,8,9,10]) #training sets index

Feature_importance_all_sets = pd.DataFrame() #initialize dataframe to save feature importance of each set
#initialize best parameters to save tuned hyperparameters for each set
n_estimators_all_sets = []
min_sample_split_all_sets = []
max_features_all_sets = []

for n in train:
    training_set = pd.read_csv(n , delimiter ='\t', index_col = 0)#Selecting our features and our target for each training set
    y_train = np.ravel(training_set[['Subgroup']])
    x_train = training_set.drop(['Subgroup'],axis=1)

    seed(500) #prevent rf randomness

    #Random Forest fhyperparameter tuning through grid search with K fold cross validation (k=5)
    parameters={'min_samples_split' : range(2,12,2), 'n_estimators': range(100,130,10), 'max_features': range(50,90,10)}
    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring = 'accuracy')
    clf.fit(x_train, y_train) #fitting the model
    print("Best parameters set found on development set:")
    print(clf.best_params_,clf.best_score_)
    min_sample_split = clf.best_params_['min_samples_split']
    n_estimators = clf.best_params_['n_estimators']
    max_features = clf.best_params_['max_features']

    #Plot features importance
    features = (clf.best_estimator_.feature_importances_)
    print(features)
    chromosomesid, importance, chromosomes = [], [], []
    for f in range(len(features)):
        if features[f] >= 0.005:
            importance.append(features[f])
            chromosomesid.append(x_train.columns[f])
            chromosomes.append(x_train.columns[f])

    #plt.bar(chromosomesid,importance), plt.xticks(rotation=90, fontsize=10)
    #plt.xlabel("Chromosome Regions"), plt.ylabel("Feature importance")
    #plt.show()
    Feature_importance = pd.DataFrame({'Chromosome id': chromosomesid, 'Importance': importance})
    Feature_importance_all_sets = pd.concat([Feature_importance_all_sets, Feature_importance], axis = 0)
    min_sample_split_all_sets.append(min_sample_split)
    max_features_all_sets.append(max_features)
    n_estimators_all_sets.append(n_estimators)


RF_CV_SUMMARY = pd.DataFrame({'Set': set, 'estimators': np.array(n_estimators_all_sets),
                                'min sample split': np.array(min_sample_split_all_sets),
                                'max features': np.array(max_features_all_sets) })

RF_CV_SUMMARY.to_csv(r'RF_CV_SUMMARY.txt', sep='\t', mode='a')

print(RF_CV_SUMMARY)
RF_CV_SUMMARY.info()


#plotting training results
max_d = RF_CV_SUMMARY[['max features']]
md, counts_md = np.unique(max_d, return_counts=True)

min_s = RF_CV_SUMMARY[['min sample split']]
ms, counts_ms = np.unique(min_s, return_counts=True)

min_l = RF_CV_SUMMARY[['estimators']]
ml, counts_ml = np.unique(min_l, return_counts=True)

plt.bar(md ,counts_md)
plt.bar(ms ,counts_ms)
plt.bar(ml ,counts_ml)
plt.show()
