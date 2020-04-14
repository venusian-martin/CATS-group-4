from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut
import numpy as np

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

classifier = NearestCentroid()
# add top feature to input array (X)
# X =
classifier.fit(X,y) #fits the NearestCentroid model to the training data
performance = classifier.score(test_X, y) # returns the accuracy of classifier.predict() for this data
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
classifier.score(testDataHere, subtypesForTestData)
