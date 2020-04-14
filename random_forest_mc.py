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


dataset = pd.read_csv('dataset_rf.csv', index_col=1)
del dataset['Unnamed: 0']
print(dataset)
y_train = dataset[['subgroup']]
x_train = dataset.drop(['subgroup'],axis=1)
#factorization
factor = pd.factorize(dataset['subgroup'])
definitions = factor[1]

labels = np.ravel(y_train)
features = pd.DataFrame(x_train).to_numpy()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


param_grid = {
    'max_depth': range(1,5),
    'min_samples_leaf': [3, 4],
    'min_samples_split': range(2,10,5)
}

rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1)

gs = grid_search.fit(train_features, train_labels)
grid_search.best_params_

best_grid = grid_search.best_estimator_

print(grid_search.best_params_,grid_search.best_score_)

y_pred = gs.predict(test_features)
print(y_pred)
rf_probs = gs.predict_proba(test_features)
print(rf_probs)

#Reverse factorize
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(test_labels)
y_pred = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

#ROC plot
curve_rf = sklearn.metrics.roc_curve(test_labels, rf_probs[:, 1])
auc_rf = auc(curve_rf[0], curve_rf[1])

plt.subplot()
plt.plot(curve_rf[0], curve_rf[1], label='Random Forest (area = %0.2f)'% auc_rf)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve');
plt.show()
