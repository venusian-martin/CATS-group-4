from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut
from statistics import mean
import numpy as np
import pandas as pd
import argparse

# parse command line arguments for input dataset and output from
parser = argparse.ArgumentParser(description='run inner loop')
parser.add_argument('-d')
args = parser.parse_args()
dir = args.d

# load ranked features
rankpath = dir + "/ranking.csv"
ranking = pd.read_csv(rankpath, index_col = 0)
# remove initial X from feature names
ranking.index = [f[1:] for f in ranking.index]
ranked_features = ranking.index

# load training and testing data, drop non-feature columns, and reorder according
# to ranking.csv
trainpath = dir + "/Train" + dir + ".txt"
train = pd.read_csv(trainpath, sep = "\t")
y = train["Subgroup"]
train = train.drop(columns = ["Unnamed: 0", "Subgroup"])
train = train.reindex(columns = ranked_features)
train = train.to_numpy()

testpath = dir + "/Test" + dir + ".txt"
test = pd.read_csv(testpath, sep = "\t")
test_y = test["Subgroup"]
test = test.drop(columns = ["Unnamed: 0", "Subgroup"])
test = test.reindex(columns = ranked_features)
test = test.to_numpy()

# initiate classifier
classifier = NearestCentroid()
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

best = performances.index(max(performances))
best_features = used_features[best]
print(best_features)

best_train = train[:,0:best_features]
best_test = test[:,0:best_features]

final_clf = classifier.fit(best_train, y)
final_predict = classifier.predict(best_test)
final_acc = classifier.score(best_test, test_y)

outpath = dir + "/predictions.csv"
np.savetxt(outpath, final_predict, fmt='%s', delimiter = ",")

with open('accuracies.txt', 'a') as f:
    line = ''.join(['Test', dir, ", ", str(final_acc), '\n'])
    f.write(str(line))
