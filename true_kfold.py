from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut
from statistics import mean
import numpy as np
import pandas as pd
import csv
import os

# import dataset
df = pd.read_csv("transposed_dataset.txt", delimiter = "\t", index_col = 0)

# import reordering of rows
order_iter = pd.read_csv("shuffled_rows_df.txt", delimiter = "\t", index_col = 0)

triplet_accuracies = []
triplet_features = []
# repeat 50 times:
for it in range(0, 50):
    # reorder rows
    rows = order_iter['iteration ' + str(it)]
    dataset = df.reindex(index = rows)

    # divide dataset into 3 parts by index
    # these indices will be dropped to get the training set
    test_indices_1 = dataset.index[0:33]
    test_indices_2 = dataset.index[33:66]
    test_indices_3 = dataset.index[66:100]

    # these indeices are kept get test sets
    test_1 = dataset.iloc[0:33]
    test_2 = dataset.iloc[33:66]
    test_3 = dataset.iloc[66:100]

    # import the rankings belonging to these splits in the dataset (range_fisher_ranking.R)
    ranking_1 = pd.read_csv(str(it+1)+"_ranking_1.csv", index_col = 0)
    ranking_2 = pd.read_csv(str(it+1)+"_ranking_2.csv", index_col = 0)
    ranking_3 = pd.read_csv(str(it+1)+"_ranking_3.csv", index_col = 0)

    # repeat for every fold
    training = [test_indices_1, test_indices_2, test_indices_3]
    testing = [test_1, test_2, test_3]
    ranks = [ranking_1, ranking_2, ranking_3]

    # to store results of each fold
    accuracies = []
    features = []
    for tr, te, r in zip(training, testing, ranks):
        # remove initial X from feature names (from loading data into Rstudio)
        ranking = r
        ranking.index = [f[1:] for f in ranking.index]
        ranked_features = ranking.index
        # drop from dataset the indices located as the test set
        train = dataset.drop(tr)
        # outcome group
        y = train["Subgroup"]
        # features only
        train = train.drop(columns = "Subgroup")
        # features reordered to match ranking
        train = train.reindex(columns = ranked_features)
        train = train.to_numpy()

        classifier = NearestCentroid()

        ### feature selection ###

        # loop initialisation
        i = 0
        cnt = 0
        increment = 5
        used_features = []
        performances = []

        # go through adding features $increment at a time
        for attempt in range(0, train.shape[1]+1, increment):
            cnt += 1

            i += increment
            used_features.append(i)
            loo = LeaveOneOut()
            X = train[:, 0:i]
            scores = []
            for train_index, test_index in loo.split(X):
                sample_train, sample_test = X[train_index], X[test_index]
                outcome_train, outcome_test = y[train_index], y[test_index]
                classifier.fit(sample_train, outcome_train)
                loo_score = classifier.score(sample_test, outcome_test)
                scores.append(loo_score)

            performance = mean(scores)
            # break when the performance has not improved in the last 5 iterations
            if cnt >= 6:
                last_5 = performances[-5:]
                # break if the performance has not improved in the last 5 iterations
                if min(last_5) >= performance:
                    break
            performances.append(performance)

        # select the number of features with the highest performance
        np.array(performances)
        #index of highest performing fit
        max_performance = np.where(performances == np.amax(performances))[0][0]

        # index 0 means 5 features, index 1 means 10 features, and so on
        best_number_of_features = increment * (max_performance+1)
        # retrieve the features used
        best_features = ranked_features[0:best_number_of_features]

        # train the model using the best features
        best_train = train[:,0:best_number_of_features]
        final_clf = classifier.fit(best_train, y)

        test_y = te["Subgroup"]
        test = te.drop(columns = "Subgroup")
        test = test.reindex(columns = ranked_features)
        test = test.to_numpy()

        best_test = test[:,0:best_number_of_features]
        # accuracy using best features
        final_acc = classifier.score(best_test, test_y)

        accuracies.append(final_acc)
        features.append(best_features)

    triplet_accuracies.append(accuracies)
    triplet_features.append(features)

# find highest performance and its index
max = np.amax(triplet_accuracies)

max_index = np.where(triplet_accuracies == max)
row_index = max_index[0][0]
col_index = max_index[1][0]

# get features for this iteration
max_features = triplet_features[row_index][col_index]

# save features to file
with open("output/features.txt", 'w') as ff:
    for f in max_features:
        ff.write(f)
        ff.write("\n")

# save all 150 accuracies to a file
accuracies = []
for trip in triplet_accuracies:
    accuracies += trip

with open("output/accuracies.txt", 'w') as af:
    for acc in accuracies:
        af.write(str(acc))
        af.write("\n")
