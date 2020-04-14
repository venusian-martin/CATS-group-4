import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV


Training_Classification = pd.read_csv('./dataset_rf.csv',index_col=1)
y_train = Training_Classification[['subgroup']]
x_train = Training_Classification.drop(['subgroup'],axis=1)

labels = np.ravel(y_train)
features = pd.DataFrame(x_train).to_numpy()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


param_grid = {
    'max_depth': range(1,20,2),
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': range(2,60,5)
}

rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1)

grid_search.fit(train_features, train_labels)
grid_search.best_params_

best_grid = grid_search.best_estimator_

print(grid_search.best_params_,grid_search.best_score_)

y_pred = grid_search.predict(test_features)
rf_probs = grid_search.predict_proba(test_features)[:, 1]
print(rf_probs)


#print(y_pred)
roc_value = roc_auc_score(test_labels, y_pred)
print(roc_value)
