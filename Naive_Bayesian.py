# Implement a machine learner using Naive Bayesian for hw3 in AI class. 

import argparse
import random
import numpy as np
import sys
import math

def main():
    #Grab the entire data set and store it in an matrix called raw_data.
    parser = argparse.ArgumentParser(description='figure out heart anomalys')
    parser.add_argument('--file', '-f', type=str, default=None, help='input file')

    args = parser.parse_args()
    if args.file == None:
        # if no data file was found return
        print("no input file detected")
        return -1
    else:
        file = args.file

    with open(args.file, 'r') as f: 
        raw_data = [[int(num) for num in line.split(",")] for line in f]
        print(np.matrix(raw_data))
        #convert matrix raw_data into an array.
        raw_data = np.array(raw_data)

    #find out the column length -1 for the first column. 
    column_length = (np.size(raw_data, 1) - 1)
    #print("column length", column_length)
    
    #generate a 2-d F matrix of column_length
    F = np.zeros((2, column_length))
    print("F matrix is: \n", F)

    #generate a 1-d P(0) matrix of column_length
    P0 = np.zeros((column_length))
    print("P0 matris is: \n", P0)

    #generate a 1-d P(1) matrix of column_length
    P1 = np.zeros((column_length))
    print("P1 matris is: \n", P1)

    #find the number of rows in the file
    rows = np.size(raw_data, 0)
    print("number of rows is: ", rows)

    #grab the first column of raw_data (aka the classify column of true/false).
    classify = raw_data[:, [0]]
    #print("classify is: \n", classify)

    #set default for abnormal and normal totals
    abnormal = 0
    normal = 0

    # loop through the number of rows we have. 
    for i in range(rows):
        # check the first row's binary. If true increment normal else increment abnormal. 
        if classify[i] == 1:
            normal = normal + 1
        else:
            abnormal = abnormal + 1

    # adding 0.5 to probability to prevent log issue later on. 
    P0_prob = (abnormal + 0.5)/(rows + 0.5)
    P1_prob = (normal + 0.5)/(rows + 0.5)

    print("normal is: ", normal)
    print("abnormal is: ", abnormal)
    print("P0_prob is: ", P0_prob)
    print("P1_prob is: ", P1_prob)



main()
