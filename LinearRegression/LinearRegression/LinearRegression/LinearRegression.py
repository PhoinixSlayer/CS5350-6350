# Main file for Linear Regression LMS homework portion.

import random as r
import math as m
import pandas as pd
# libraries for ease of convenience, used to create graphs with the collected data
import matplotlib as mat
import matplotlib.pyplot as pyp



class Example:
    def __init__(self, ex_number, attributes, slump):
        self.example_number = ex_number
        self.attributes = attributes
        self.y = slump


# Calculate the cost for a specific examples mistake
def Cost(weight_vector, example):
    vector_mult = 0
    for f in example.attributes:
        vector_mult += example.attributes[f] * weight_vector[f]
    cost = example.slump - vector_mult

# Calculate the total Cost (or Loss) for the passed in weight vector using the data
def Loss(weight_vector, examples):
    sum = 0
    for ex in examples:
        example = examples[ex]
        cost_ex = Cost(weight_vector, example)
        sum += cost_ex**2
    return sum / 2

# Calculates the gradient for the given part of J()'s derivative xi using the passed in weight and example
def GradientPart(weight_vector, example, xi):
    cost_ex = Cost(weight_vector, example)
    return -(cost_ex * xi)

# Returns total gradient for the weight vector as a new vector.


def main():
    #bank_train_labels = Label_data(0, {"yes": 0, "no": 0})
    #bank_test_labels = Label_data(0, {"yes": 0, "no": 0})

    train_examples = {}
    test_examples = {}
    
    id = 1
    with open ( "concrete/train.csv" , 'r' ) as f: 
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], terms[6] ]
            temp = Example(id, attributes, terms[7]) 
            train_examples.update({id: temp})
            id += 1
            
    id = 1
    with open( "concrete/test.csv", 'r') as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], terms[6] ]
            temp = Example(id, attributes, terms[7])
            test_examples.update({id: temp})
            id += 1

    base_weight = [0, 0, 0, 0, 0, 0, 0, 0]
    rate = 1

    weight_difference = 1
    while weight_difference > 10**(-6):
        new_weight = 0



# Run program
main()