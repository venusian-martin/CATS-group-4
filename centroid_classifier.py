from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import argparse

''' this was an copied from an example in the documentation for me to see the data types etc

from sklearn import datasets, neighbors, linear_model

X_digits, y_digits = datasets.load_digits(return_X_y=True)
X_digits = X_digits / X_digits.max()

print(type(X_digits), X_digits)
print(type(y_digits), y_digits)
'''

''' this part might be better to do in R since the fisher test is much easier to perform

# open the arrayCGH data file
# append each line as row in numpy array (all_samples)
# split into train_samples and test_samples
# shuffle() will randomise the order so we get a different 80/20 split each time

# perform fisher exact tests on each feature (region in each sample)
# the values of 1 feature for every train_sample is simply that column
# rank features based on fisher test score

# open the subtype data file (training data part)
# split into train_y and test_y (make sure the split is the same as the samples!)

in that case it would just be:

# load training arrayCGH data
# load testing arrayCGH data
# load training subtype data
# load testing subtype data
# load ranked list of features
'''
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
print("training data: ", train)

testpath = dir + "/Test" + dir + ".txt"
test = pd.read_csv(testpath, sep = "\t")
test_y = test["Subgroup"]
test = test.drop(columns = ["Unnamed: 0", "Subgroup"])
test = test.reindex(columns = ranked_features)
test_X = test.to_numpy()


#print(test)
classifier = NearestCentroid()

# add top feature to numpy array (X)
X = train[:,0]
X = X.reshape(-1, 1)
classifier.fit(X,y) #fits the NearestCentroid model to the training data
print("I DID THING")
print("X shape: ", test_X.shape, "y shape: ", test_y.shape)
print(type(test_X), type(test_y))
performance = classifier.score(test_X, test_y) # returns the accuracy of classifier.predict() for this data
previous_performance = 0 #(placeholder so the loop will work)

while performance > previous_performance:
    previous_performance = performance
    X += next_top_feature(s) #need to decide how to do this. add 1 each time? add 5? add
    loo = LeaveOneOut() # function to compute the indices which split the data so that each sample is used for testing once
    scores = [] # will store the score of each predicition
    for train_index, test_index in loo.split(X): # this method gives returns the indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scores.append(score)
        performance = mean(scores)

# now get accuracy for the test data set aside earlier
accuracy = classifier.score(testDataHere, subtypesForTestData)
