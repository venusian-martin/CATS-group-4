#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
# Start your coding
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut
from statistics import mean
import numpy as np
import pandas as pd
import csv
import pickle
# import the library you need here

# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # suggested steps
    pkl = args.model_file
    # Step 1: load the model from the model file
    infile = open(pkl, 'rb')
    model_and_features = pickle.load(infile)
    model = model_and_features[0]
    features = model_and_features[1]
    # Step 2: apply the model to the input file to do the prediction
    datafile = args.input_file
    data = pd.read_csv(datafile, delimiter = "\t", index_col = 0)
    data = data.drop(["Nclone", "Start", "End"], axis = 1)
    data = data.transpose()
    print(list(data))
    data = data.loc[: , features]
    prediction = model.predict(data)
    # Step 3: write the prediction into the desinated output file
    outfile = args.output_file
    with open(outfile, 'w') as of:
        for p in prediction:
            of.write(p, '\n')
    # End your coding


if __name__ == '__main__':
    main()
