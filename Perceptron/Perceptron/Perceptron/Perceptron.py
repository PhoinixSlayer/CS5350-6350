# Main file for Perceptron assignment.

import random as r
import math as m
import pandas as pd
# libraries for ease of convenience, used to create graphs with the collected data
import matplotlib as mat
import matplotlib.pyplot as pyp


class Example:
    def __init__(self, ex_number, attributes, labl):
        self.example_number = ex_number
        self.attributes = attributes
        self.label = labl



def main():
    train_examples = {}
    test_examples = {}
    
    id = 1
    with open ( "bank-note/train.csv" , 'r' ) as f: 
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, terms[0], terms[1], terms[2], terms[3] ]
            temp = Example(id, attributes, terms[4])  # Label of example, if 0 means forged, 1 means genuine
            train_examples.update({id: temp})
            id += 1
            
    id = 1
    with open( "bank-note/test.csv", 'r') as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, terms[0], terms[1], terms[2], terms[3] ]
            temp = Example(id, attributes, terms[4])
            test_examples.update({id: temp})
            id += 1

    ### Perceptron ###
    T = 10

    StandardPerceptron(train_examples, test_examples, T)

    VotedPerceptron(train_examples, test_examples, T)

    AveragePerceptron(train_examples, test_examples, T)
    
    ### Not sure how I'll do it yet, but need to gather the average prediction error for each of the three and then compare them.

    # Compare the average prediction errors for each of the methods above