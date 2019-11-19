Bar Movshovich
CS541
https://github.com/movshov/heart_anomaly
# Heart Anomaly
Given a set of data like the one shown below using Naive Baysian Learning and K-Nearest Neighbor predict whether a person has a normal or abnormal heart.  

Example of spect-orig.train.csv that we will use to train our learners: 
```
1,0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0
1,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1
1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0
1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1
1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0
1,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1
1,1,0,1,1,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,1
1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1
1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1
1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0
1,1,1,0,0,1,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,0,1
1,1,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,1,0,1,1,0,0
                     ...
```
The first column represents the final verdict on whether this heart is normal(1) or abnormal(0). The second column all the way to the last column is a "feature" with a 1 representing true and 0 representing false. Using Naive Bayesian and K-Nearest Neighbor we will "train" our learners on this training data before checking our prediction against a test csv file like the one shown below: 

Example of spect-orig.test.csv file: 
```
1,1,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,0
1,1,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0
1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0,1
1,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,1,0,0,0,0,1,0
1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,0,0,1
1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,1
1,1,0,0,1,0,0,1,1,1,1,0,1,1,1,0,1,0,0,0,1,0,1
1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0
1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0
1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0,0,0
1,1,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0
1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0
1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1
1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0
1,1,0,0,1,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,0
1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1,0,0,1,0
1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,1,0,0,0,1,1
                  ...
```

# Build
This program has two different build paths with several flags that need to be set.  

The first build example is how to run the program using the Naive_Bayesian.py file.
```
python3 Naive_Bayesian.py --help
usage: Naive_Bayesian.py [-h] [--train TRAIN] [--test TEST]

figure out heart anomalys

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN, -tr TRAIN
                        train file
  --test TEST, -te TEST
                        test file
```

To run the program run the following command: 
```
python3 Naive_Bayesian.py -tr <YOUR TRAIN FILE HERE> -te <YOUR TEST FILE HERE>
```
You will need to replace <YOUR TRAIN FILE HERE> with whatever file you want to use for your train set and <YOUR TEST FILE HERE> with whatever test file you wish to use.

For example if you wanted to use the spect-orig files you would type: 
```
python3 Naive_Bayesian.py -tr spect-orig.train.csv -te spect-orig.test.csv
```
This will run the program using Naive Bayesian learning with spect-orig.train.csv file as the train file and spect-orig.test.csv as the test file. The program will then run and will display quite a few pieces of information for debugging purposes. These print statments can be turned off but are left on for clarity. The result of the program should look like the following: 
```
raw_data of file is: 
 [[1 0 0 ... 0 0 0]
 [1 0 0 ... 0 0 1]
 [1 1 0 ... 0 0 0]
 ...
 [0 1 0 ... 0 0 0]
 [0 0 0 ... 0 1 1]
 [0 1 0 ... 0 0 0]]
number of rows is:  80
normal is:  40
abnormal is:  40
P0_prob is:  0.5031055900621118
P1_prob is:  0.5031055900621118
test_data of file is: 
 [[1 1 0 ... 1 0 0]
 [1 1 0 ... 0 0 0]
 [1 0 0 ... 0 0 1]
 ...
 [0 1 0 ... 0 0 0]
 [0 1 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
test_rows_length is:  187
Naive_guess is: 
 [1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1.
 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 0. 1. 0. 1.
 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 1.
 0. 0. 0. 1. 1. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 0. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 1. 1. 0.]
file: spect-orig.train.csv 142/187 (0.7593582887700535)
10/15(0.6666666666666666)
132/172(0.7674418604651163)
```

The first matrix shows the original train data followed by the number of rows it contains, the number of normal hearts in the first column (nomral), the number of abnormal hearts in the first column (abnormal), the probability of an abnormal heart based on the first column (P0_prob), and the probability of a normal heart based on the first column (P1_prob). 

Next we have the matrix for the test file followed by the number of rows in the test file and then my Naive Bayesian Guess array. 

Finally, we have the file name followed by the fraction of instances that were classified correctly. Then the second number is the fraction of abnormal instances that were classified correctly. Lastly, the third number is the fractoin of normal instances that were classified correctly. 

To run the K-Nearest Neighbor version of this code you will need to add a new flag to your run command "k". Information about the new flag can be found by running the following command: 

```
python3 K_Nearest_Neighbor.py --help
usage: K_Nearest_Neighbor.py [-h] [--train TRAIN] [--test TEST] [--k K]

figure out heart anomalys

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN, -tr TRAIN
                        train file
  --test TEST, -te TEST
                        test file
  --k K, -k K           length of k for KNN
 ```


To run K-Nearest Neighbor run the following command: 
```
python3 K_Nearest_Neighbor.py -tr <YOUR TRAIN FILE HERE> -te <YOUR TEST FILE HERE> -k <YOUR K NUMBER>
```
You will need to replace <YOUR TRAIN FILE HERE> with whatever file you want to use for your train set and <YOUR TEST FILE HERE> with whatever test file you wish to use. K will need to be some integer. Ideally only use ODD numbers to avoid a tie.

An example would be as follows: 
```
python3 K_Nearest_Neighbor.py -tr spect-orig.train.csv -te spect-orig.test.csv -k 5
```
This code will print the results in the exact same format as previously mentioned above.

# Algorithm

