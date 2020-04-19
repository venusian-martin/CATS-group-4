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
from sklearn.metrics import confusion_matrix

RF_summary = pd.read_csv('RF_CV_SUMMARY.txt', delimiter = '\t', index_col = 0)
print(RF_summary)


train = ['Train1.txt', 'Train2.txt', 'Train3.txt', 'Train4.txt', 'Train5.txt',
            'Train6.txt', 'Train7.txt', 'Train8.txt','Train9.txt','Train10.txt'] #our training sets
test = ['Test1.txt', 'Test2.txt', 'Test3.txt', 'Test4.txt', 'Test5.txt',
            'Test6.txt', 'Test7.txt', 'Test8.txt','Test9.txt','Test10.txt'] #our test sets

set = np.array([1,2,3,4,5,6,7,8,9,10]) #test sets index

Feature_importance_all_sets = pd.DataFrame() #initialize dataframe to save feature importance of each set
Predictions = pd.DataFrame() #initialize dataframe to save predictions
Accuracy_score = [] #initialize list to append accuracy per set

for n,m in zip(train,test):
    training_set = pd.read_csv(n , delimiter ='\t', index_col = 0)#Selecting our features and our target for each training set
    test_set = pd.read_csv(m , delimiter ='\t', index_col = 0)#Selecting our features and our target for each test set
    y_train = np.ravel(training_set[['Subgroup']])
    x_train = training_set.drop(['Subgroup'],axis=1)
    y_test = np.ravel(test_set[['Subgroup']])
    x_test = test_set.drop(['Subgroup'],axis=1)
    print(n)
    #getting hyperparameters for each set
    min_sample = RF_summary.at[train.index(n), 'min sample split']
    n_estimators = RF_summary.at[train.index(n), 'estimators']
    max_features = RF_summary.at[train.index(n), 'max features']

    seed(500) #prevent rf randomness

    #Random Forest fhyperparameter tuning through grid search with K fold cross validation (k=5)
    parameters={'min_samples_split': [min_sample] ,'max_features': [max_features] , 'n_estimators': [n_estimators]}
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

    Feature_importance = pd.DataFrame({'Chromosome id': chromosomesid, 'Importance': importance})
    Feature_importance_all_sets = pd.concat([Feature_importance_all_sets, Feature_importance], axis = 1)


    #Predicting
    y_pred = clf.predict(x_test)
    predictions_test = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()}) #dataframe with predictions vs actual
    ACC_test = accuracy_score(np.array(y_test), np.array(y_pred))
    Accuracy_score.append(ACC_test)
    Predictions = pd.concat([Predictions, predictions_test], axis = 1) #saving predictions of each test


RF_ACCURACY = pd.DataFrame({'Set': set, 'Accuracy': Accuracy_score})
print(RF_ACCURACY)
print(Feature_importance_all_sets)
print(Predictions)


#Look for maximum accuracy in RF_ACCURACY
index_max_acc = RF_ACCURACY['Accuracy'].idxmax()
index_max_acc = index_max_acc + 1
print('Best performing set is: {}'.format(index_max_acc))
print('Best performing set accuracy is: {}'.format(RF_ACCURACY['Accuracy'].max()))

position = 2*index_max_acc - 2 #position of feature importance for this set in the feature importance dataframe
chr_id = Feature_importance_all_sets.iloc[:,position] #chromosomes for the best set
chr_im = Feature_importance_all_sets.iloc[:,position+1] #chr importance for the best set
chr_id = chr_id.dropna()
chr_im = chr_im.dropna()
#plot feature importance
plt.bar(chr_id,chr_im), plt.xticks(rotation=45, fontsize=10)
plt.xlabel("Chromosome Regions"), plt.ylabel("Feature importance")
plt.show()


actual = Predictions.iloc[:,position] #real labels for the test set
predicted = Predictions.iloc[:,position+1] #predictions for the best set

#confusion matrix
cnf_matrix = confusion_matrix(np.ravel(actual), np.ravel(predicted), labels=['HER2+', 'HR+', 'Triple Neg'])
print(cnf_matrix)

df_cm = pd.DataFrame(cnf_matrix, range(3), range(3))
plt.figure(figsize=(10,7))
sns.set(font_scale=1) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 9}) # font size
plt.xticks([0,1,2],['HER2+', 'HR+', 'Triple Neg']); plt.yticks([0,1,2],['HER2+', 'HR+', 'Triple Neg']);
plt.show()
