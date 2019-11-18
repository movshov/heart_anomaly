# Implement a machine learner using Naive Bayesian for hw3 in AI class. 

import argparse
import random
import numpy as np
import sys
import math

def main():
    # Grab the entire data set and store it in an matrix called raw_data.
    parser = argparse.ArgumentParser(description='figure out heart anomalys')
    parser.add_argument('--train', '-tr', type=str, default=None, help='train file')
    parser.add_argument('--test', '-te', type=str, default=None, help='test file')
    parser.add_argument('--k', '-k', type=int, default=1, help='length of k for KNN')

    args = parser.parse_args()
    if args.train == None or args.test == None:
        # if no train or test file was found return
        print("no input file detected")
        return -1
    raw_data = None
    if args.k % 2 == 0:
        k = args.k + 1
    else:
        k = args.k
    with open(args.train, 'r') as f: 
        raw_data = [[int(num) for num in line.split(",")] for line in f]
        print("raw_data of file is: \n", np.matrix(raw_data))
        #print("args.train is: ", args.train)
        #convert matrix raw_data into an array.
        raw_data = np.array(raw_data)
        f.close()

    # find out the column length -1 for the first column. 
    column_length = (np.size(raw_data, 1) - 1)
    # print("column length", column_length)
    
    # generate a 2-d F matrix of column_length # F = np.zeros((2, column_length))
    # print("F matrix is: \n", F)

    # generate a 1-d P(0) matrix of column_length for every possible comb.
    P0_0 = np.zeros((column_length))
    P0_1 = np.zeros((column_length))
    P1_0 = np.zeros((column_length))
    P1_1 = np.zeros((column_length))

    # find the number of rows in the file
    rows = np.size(raw_data, 0)
    print("number of rows is: ", rows)

    # grab the first column of raw_data (aka the classify column of true/false).
    classify = raw_data[:, [0]]
    # print("classify is: \n", classify)

    # Now that we have P(0) and P(1) we now need to find the probability of each column. 
    # we start at the 2nd column. 
    for i in range(column_length):
        # expect abnormal, get abnormal
        temp_P0_0 = 0
        # expect abnormal, get normal
        temp_P0_1 = 0
        # expect normal, get abnormal
        temp_P1_0 = 0
        # expect normal, get normal
        temp_P1_1 = 0

        # grab the ith column +1 to avoid 1st column
        column = raw_data[:, [i+1]]
        # print("other column is: ", column) 

        # Loop through every row of length "k" and check if the test is a 1 or 0. 
        for j in range(rows):
            # loop through K nearest neighbors.
            for d in range(k):
                # make sure we don't go out of bounds.
                if j <= (rows-k):
                    if column[j+d] == 1:
                        if classify[j+d] == 1: 
                            temp_P1_1 += 1
                        else: 
                            temp_P1_0 += 1

                    elif column[j+d] == 0:
                        if classify[j+d] == 1: 
                            temp_P0_1 += 1
                        else: 
                            temp_P0_0 += 1

        # get the KNN distance for each probability.
        P0_0[i] = temp_P0_0
        P1_0[i] = temp_P1_0
        P1_1[i] = temp_P1_1
        P0_1[i] = temp_P0_1
        temp_P0_0 = 0
        temp_P0_1 = 0
        temp_P1_0 = 0
        temp_P1_1 = 0

    # probability based off of training data.
    print("P0_0 is \n", P0_0)
    print("P1_0 is \n", P1_0)
    print("P1_1 is \n", P1_1)
    print("P0_1 is \n", P0_1)

    # open up test data and save to array called test_data 
    with open(args.test, 'r') as k: 
        test_data = [[int(num) for num in line.split(",")] for line in k]
        print("test_data of file is: \n", np.matrix(test_data))
        #convert matrix raw_data into an array.
        test_data = np.array(test_data)

    # grab test_data column length but skipping 1 column. 
    test_column_length = (np.size(test_data, 1) - 1)
    # print("test_column_length is: ", test_column_length)
    test_rows_length = np.size(test_data, 0)
    print("test_rows_length is: ", test_rows_length)

    # Final result.
    KNN_guess = np.zeros((test_rows_length))

    # Loop through all test_rows for length of "k"
    for j in range(test_rows_length):
        test_rows = test_data[[j],:]
        #print("test_rows is: ", test_rows)
        # Loop through all test_columns
        P_AB = 0
        P_N = 0
        for i in range(test_column_length):
            # we add 1 to skip first column
            # print("i is: ", i)
            if test_rows[0][i+1] == 0:
                P_AB += P0_0[i]/(P0_0[i] + P0_1[i])
                P_N += P0_1[i]/(P0_1[i] + P0_0[i])

            # we add 1 to skip first column
            elif test_rows[0][i+1] == 1:
                P_AB += P1_0[i]/(P1_0[i] + P1_1[i])
                P_N += P1_1[i]/(P1_1[i] + P1_0[i])


        # print("P_AB is: \n", P_AB)
        # print("P_N is: \n", P_N)
        if P_AB > P_N:
            KNN_guess[j] = 0
        else:
            KNN_guess[j] = 1

    print("KNN_guess is: \n", KNN_guess)

    test_column = test_data[:, [0]]
    # print("test_column is: \n", test_column)

    classified_correctly = 0 
    abnormal_correctly = 0
    normal_correctly = 0
    abnormal_count = 0
    normal_count = 0

    for i in range(test_rows_length):
        if (test_column[i] == 0):
            abnormal_count += 1
        if (test_column[i] == 1):
            normal_count += 1
        if (test_column[i] == 1 and KNN_guess[i] == 1):
            classified_correctly += 1
            normal_correctly += 1
        elif (test_column[i] == 0 and KNN_guess[i] == 0):
            classified_correctly += 1
            abnormal_correctly += 1

    print("file: {} {}/{} ({})".format(args.train, classified_correctly, test_rows_length, classified_correctly/test_rows_length))
    print("{}/{}({})".format(abnormal_correctly, abnormal_count, abnormal_correctly/abnormal_count))
    print("{}/{}({})".format(normal_correctly, normal_count, normal_correctly/normal_count))

main()










