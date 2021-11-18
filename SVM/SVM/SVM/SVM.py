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


# If need to update weight, first multiply each value of e's feature vector by e's label value and the rate. Then add that to the weight.
def Update(w, g, C, N, example): #! Need to change soon
    new_weight = [0] * len(w)

    for i in len(w):
        new_weight[i] = w[i] - g*w[i] + g * C * N * example.label * example.attributes[i]

    return new_weight

## I think this is fine because we do the SVM version of updating if this is less than 0
def VectorMult(weight, example):
    total = 0
    for i in range(len(weight)):
        total += example.attributes[i] * weight[i]
    total += total * example.label
    return total

##
def Predict(weight, example):
    total = 0
    for i in range(len(weight)):
        total += example.attributes[i] * weight[i]
    return total


## Need to figure out what the 'a' variable is
def Schedule1(g, a, t):
    return (g/(1+g/a*t))

def Schedule2(g, t):
    return (g/(1 + t))


def SVM_Stochastic(train, test, T):
    w1, w2, w3 = [0,0,0,0,0]
    N = len(train)
    gamma = 1  # In this algorithm this is the learning rate
    curr_gamma = 1
    a = 1 ## I don't know what this is yet, really it's just a place holder.
    t = 0

    vectors = []  # Will I need something like this?
    vectors.append(w)

    for i in range(T):
        shuffled_order = r.sample( range(1, len(train)+1), len(train))
        for index in shuffled_order:
            example = train[index]
            pred1 = VectorMult(w1, example)
            pred2 = VectorMult(w2, example)
            pred3 = VectorMult(w3, example)
            if pred1 <= 1:
                w1 = Update(w1, curr_gamma, C[0], N, example)
            else:
                for j in len(w1):
                    w1[j] = (1-curr_gamma) * w1[j]

            if pred2 <= 1:
                w2 = Update(w2, curr_gamma, C[1], N, example)
            else:
                for j in len(w2):
                    w1[j] = (1-curr_gamma) * w1[j]

            if pred3 <= 1:
                w3 = Update(w3, curr_gamma, C[2], N, example)
            else:
                for j in len(w3):
                    w3[j] = (1-curr_gamma) * w3[j]

            t += 1 # Do this here? or after?
            curr_gamma = Schedule1(gamma, a, t)

            #they want me to tune the original gamma and a, does this mean I decrease those separately along with the one we're using for
            # this iteration? Or do we set this to the original gamma, then update it for each example, then when we reset for the next
            # epoch we set the main gamma to something less with a simple calculation, so this iterated gamma based on t is different
            # each epoch?

    # With the weight vectors complete, we can now run tests with them
    error1, error2, error3 = 0  # For now we'll simply tally incorrect predictions on the data
    for i in train:
        example = train[i]
        result = Test(w1, example)
        sign = 0
        if result >= 0:
            sign = 1
        else:
            sign = -1
        if sign != example.label:
            error1 += 1

        result = Test(w2, example)
        sign = 0
        if result >= 0:
            sign = 1
        else: 
            sign = -1
        if sign != example.label:
            error2 += 1

        result = Test(w3, example)
        sign = 0
        if result >= 0:
            sign = 1
        else:
            sign = -1
        if sign != example.label:
            error3 += 1

    error1 = error1 / len(train)
    error2 = error2 / len(train)
    error3 = error3 / len(train)

    terr1, terr2, terr3 = 0
    for i in test:
        example = test[i]
        result = Test(w1, example)
        sign = 0
        if result >= 0:
            sign = 1
        else:
            sign = -1
        if sign != example.label:
            terr1 += 1

        result = Test(w2, example)
        sign = 0
        if result >= 0:
            sign = 1
        else: 
            sign = -1
        if sign != example.label:
            terr2 += 1

        result = Test(w3, example)
        sign = 0
        if result >= 0:
            sign = 1
        else:
            sign = -1
        if sign != example.label:
            terr3 += 1

    terr1 = terr1 / len(test)
    terr2 = terr2 / len(test)
    terr3 = terr3 / len(test)


def DualDomain(train, test, T):
    w = [0,0,0,0,0]
    N = len(train)
    gamma = 1
    for i in range(T):
        shuffled_order = r.sample( range(1, len(train)+1), len(train))


C = [100/873, 500/873, 700/873]

###
def main():
    train_examples = {}
    test_examples = {}
    
    id = 1
    label = 1
    with open ( "bank-note/train.csv" , 'r' ) as f: 
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3]) ]
            if float(terms(4)) == 0:
                label = -1
            else:
                label = 1
            temp = Example(id, attributes, label)  # Label of example, if 0 means forged, 1 means genuine
            train_examples.update({id: temp})
            id += 1
            
    id = 1
    label = 1
    with open( "bank-note/test.csv", 'r') as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3]) ]
            if float(terms(4)) == 0:
                label = -1
            else:
                label = 1 
            temp = Example(id, attributes, label)
            test_examples.update({id: temp})
            id += 1


    SVM_Stochastic(train_examples, test_examples, 100)

    DualDomain(train_examples, test_examples, 100)



####################################################################################################################
main()
