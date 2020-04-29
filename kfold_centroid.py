from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut
from statistics import mean
import numpy as np
import pandas as pd
import csv

dataset = pd.read_csv("transposed_dataset.txt", delimiter = "\t", index_col = 0)

#dividing into 3 equal parts
train_1 = dataset.index[0:33]
train_2 = dataset.index[33:66]
train_3 = dataset.index[66:100]

test_1 = dataset.iloc[0:33]
test_2 = dataset.iloc[33:66]
test_3 = dataset.iloc[66:100]

# these files are made by R script range_fisher_ranking.R
ranking_1 = pd.read_csv("ranking_1.csv", index_col = 0)
ranking_2 = pd.read_csv("ranking_2.csv", index_col = 0)
ranking_3 = pd.read_csv("ranking_3.csv", index_col = 0)

training = [train_1, train_2, train_3]
testing = [test_1, test_2, test_3]
ranks = [ranking_1, ranking_2, ranking_3]

accuracy = []
predictions = []

for tr, te, r in zip(training, testing, ranks):
    # remove initial X from feature names (not found in pandas dataset)
    ranking = r
    ranking.index = [f[1:] for f in ranking.index]
    ranked_features = ranking.index

    # remove rows defined as test set from dataset for training
    train = dataset.drop(tr)
    # outcome group
    y = train["Subgroup"]
    # features only
    train = train.drop(columns = "Subgroup")
    # features reordered to match ranking
    train = train.reindex(columns = ranked_features)
    train = train.to_numpy()

    # do same reordering for test set
    test_y = te["Subgroup"]
    test = te.drop(columns = "Subgroup")
    test = test.reindex(columns = ranked_features)
    test = test.to_numpy()

    classifier = NearestCentroid()

    # loop initialisation
    i = 0
    cnt = 0
    increment = 5
    used_features = []
    performances = []

    # feature selection
    for attempt in range(0, train.shape[1] +1, increment):
        cnt += 1
        i += increment # how many features to add each time
        used_features.append(i)
        X = train[:,0:i]
        loo = LeaveOneOut() # function to compute the indices which split the data so that each sample is used for testing once
        scores = []

        for train_index, test_index in loo.split(X): # this method gives the indices to use each sample as the test once
            X_train, X_test = X[train_index], X[test_index]
            #print(X_train)
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)
            score = classifier.score(X_test, y_test)
            scores.append(score)

        performance = mean(scores)

        if cnt >= 6:
            last_5 = performances[-5:]
            # break if the performance has not improved in the last 5 iterations
            if min(last_5) >= performance:
                break
        performances.append(performance)

    # find model with best performance and extract the features used
    f_cnt = performances.index(max(performances))
    best_features = used_features[f_cnt]

    best_train = train[:,0:best_features]
    best_test = test[:,0:best_features]

    # train classifier on optimal number of features
    final_clf = classifier.fit(best_train, y)
    # final prediction
    final_predict = classifier.predict(best_test)
    final_acc = classifier.score(best_test, test_y)
    accuracy.append(final_acc)
    predictions.append(final_predict)

print(len(predictions))
print(type(predictions))
# save outputs
with open("predictions.csv", 'w') as f:
    wr = csv.writer(f)
    wr.writerows(predictions)

np.savetxt("accuracies.csv", accuracy, fmt = "%s", delimiter = ",")
