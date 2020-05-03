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

all_mean_accuracies = []
fold_accuracies = []
fold_features = []
# repeat 50 times:
for it in range(0, 4):
    # reorder rows
    rows = order_iter['iteration ' + str(it)]
    dataset = df.reindex(index = rows)

    # divide dataset into 3 parts by index
    # these indices will be dropped to get the training set
    test_indices_1 = dataset.index[0:33]
    test_indices_2 = dataset.index[33:66]
    test_indices_3 = dataset.index[66:100]

    # slices of the dataframe to get test sets
    test_1 = dataset.iloc[0:33]
    test_2 = dataset.iloc[33:66]
    test_3 = dataset.iloc[66:100]
    # import the rankings belonging to these splits in the dataset (range_fisher_ranking.R)
    ranking_1 = pd.read_csv(str(it+1)+"_ranking_1.csv", index_col = 0)
    ranking_2 = pd.read_csv(str(it+1)+"_ranking_2.csv", index_col = 0)
    ranking_3 = pd.read_csv(str(it+1)+"_ranking_3.csv", index_col = 0)

    # repeat for every fold (3)
    training = [test_indices_1, test_indices_2, test_indices_3]
    testing = [test_1, test_2, test_3]
    ranks = [ranking_1, ranking_2, ranking_3]


    for tr, te, r in zip(training, testing, ranks):

        # remove initial X from feature names (not found in pandas dataset)
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

        # feature selection
        features_fold = []
        performances_fold = []

        # loop initialisation
        i = 0
        cnt = 0
        increment = 5
        used_features = []
        performances = []

        # go through adding features (increment) at a time
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
        print(performances)
        np.array(performances)
        #index of highest performing fit
        max_performance = np.where(performances == np.amax(performances))[0][0]
        print(type(max_performance))
        print(max_performance)
        best_number_of_features = increment * (max_performance+1)


        best_train = train[:,0:best_number_of_features]
        final_clf = classifier.fit(best_train, y)

        test_y = te["Subgroup"]
        test = te.drop(columns = "Subgroup")
        test = test.reindex(columns = ranked_features)
        test = test.to_numpy()

        best_test = test[:,0:best_number_of_features]
        # final prediction
        final_predict = classifier.predict(best_test)
        final_acc = classifier.score(best_test, test_y)

        fold_accuracies.append(final_acc)
        fold_features.append(ranked_features[0:best_number_of_features])

print(fold_accuracies)
print(fold_features)

# save this accuracy to file
