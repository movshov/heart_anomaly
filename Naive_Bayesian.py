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

    args = parser.parse_args()
    if args.train == None or args.test == None:
        # if no train or test file was found return
        print("no input file detected")
        return -1
    raw_data = None
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

    # set default for abnormal and normal totals
    abnormal = 0
    normal = 0

    # loop through the number of rows we have. 
    for i in range(rows):
        # check the first row's binary. If true increment normal else increment abnormal. 
        if classify[i] == 1:
            normal += 1
        elif classify[i] == 0:
            abnormal += 1

    # adding 0.5 to probability to prevent log issue later on. 
    P0_prob = (abnormal + 0.5)/(rows + 0.5)
    P1_prob = (normal + 0.5)/(rows + 0.5)

    # print out the number of normal and abnormal as well as 
    # the probability of each. 
    print("normal is: ", normal)
    print("abnormal is: ", abnormal)
    print("P0_prob is: ", P0_prob)
    print("P1_prob is: ", P1_prob)


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

        # Loop through every row and check if the test is a 1 or 0. 
        for j in range(rows):
            if column[j] == 1:
                if classify[j] == 1: 
                    temp_P1_1 += 1
                else: 
                    temp_P1_0 += 1

            elif column[j] == 0:
                if classify[j] == 1: 
                    temp_P0_1 += 1
                else: 
                    temp_P0_0 += 1

        # add the probability of 0 to P0 at index i
        P0_0[i] = np.log2((temp_P0_0 + 0.5)/(abnormal + 0.5))
        P1_0[i] = np.log2((temp_P1_0 + 0.5)/(abnormal + 0.5))
        P1_1[i] = np.log2((temp_P1_1 + 0.5)/(normal + 0.5))
        P0_1[i] = np.log2((temp_P0_1 + 0.5)/(normal + 0.5))

    # probability based off of training data.
    # print("P0_0 is \n", P0_0)
    # print("P1_0 is \n", P1_0)
    # print("P1_1 is \n", P1_1)
    # print("P0_1 is \n", P0_1)

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
    Naive_guess = np.zeros((test_rows_length))

    # Loop through all test_rows
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
                P_AB += P0_0[i]
                P_N += P0_1[i]

            # we add 1 to skip first column
            elif test_rows[0][i+1] == 1:

                P_AB += P1_0[i]
                P_N += P1_1[i]

        P_AB += P0_prob
        P_N += P1_prob

        if P_AB > P_N:
            Naive_guess[j] = 0
        else:
            Naive_guess[j] = 1

    print("Naive_guess is: \n", Naive_guess)

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
        if (test_column[i] == 1 and Naive_guess[i] == 1):
            classified_correctly += 1
            normal_correctly += 1
        elif (test_column[i] == 0 and Naive_guess[i] == 0):
            classified_correctly += 1
            abnormal_correctly += 1

    print("file: {} {}/{} ({})".format(args.train, classified_correctly, test_rows_length, classified_correctly/test_rows_length))
    print("{}/{}({})".format(abnormal_correctly, abnormal_count, abnormal_correctly/abnormal_count))
    print("{}/{}({})".format(normal_correctly, normal_count, normal_correctly/normal_count))

main()










