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

    for i in range(len(w)):
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
def Test(weight, example):
    total = 0
    for i in range(len(weight)):
        total += example.attributes[i] * weight[i]
    return total


## Need to figure out what the 'a' variable is
def Schedule1(g, a, t):
    return (g/(1+g/a*t))

def Schedule2(g, t):
    return (g/(1 + t))


def CompSign(res):
    sign = 0
    if res >= 0:
        sign = 1
    else:
        sign = -1
    return sign


def SVM_Stochastic(train, test, T):
    w1 = [0,0,0,0,0]
    w2 = [0,0,0,0,0]
    w3 = [0,0,0,0,0]
    ww1 = [0,0,0,0,0]
    ww2 = [0,0,0,0,0]
    ww3 = [0,0,0,0,0]
    N = len(train)
    gamma = 1  # In this algorithm this is the learning rate
    curr_gamma = 1
    a = 1 ## I don't know what this is yet, really it's just a place holder.
    t = 0

    for i in range(T):
        shuffled_order = r.sample( range(1, len(train)+1), len(train))
        for index in shuffled_order:
            example = train[index]
            pred1 = VectorMult(w1, example)
            if pred1 <= 1:
                w1 = Update(w1, curr_gamma, C[0], N, example)
            else:
                for j in range(len(w1)):
                    w1[j] = (1-curr_gamma) * w1[j]

        shuffled_order = r.sample( range(1, len(train)+1), len(train))
        for index in shuffled_order:
            example = train[index]
            pred2 = VectorMult(w2, example)
            if pred2 <= 1:
                w2 = Update(w2, curr_gamma, C[1], N, example)
            else:
                for j in range(len(w2)):
                    w2[j] = (1-curr_gamma) * w2[j]

        shuffled_order = r.sample( range(1, len(train)+1), len(train))
        for index in shuffled_order:
            example = train[index]
            pred3 = VectorMult(w3, example)
            if pred3 <= 1:
                w3 = Update(w3, curr_gamma, C[2], N, example)
            else:
                for j in range(len(w3)):
                    w3[j] = (1-curr_gamma) * w3[j]

        t += 1 # Do this here? or after?
        curr_gamma = Schedule1(gamma, a, t)
        a = a * 0.99

            #they want me to tune the original gamma and a, does this mean I decrease those separately along with the one we're using for
            # this iteration? Or do we set this to the original gamma, then update it for each example, then when we reset for the next
            # epoch we set the main gamma to something less with a simple calculation, so this iterated gamma based on t is different
            # each epoch?

    N = len(train)
    gamma = 1  # In this algorithm this is the learning rate
    curr_gamma = 1
    a = 1
    t = 0
    for i in range(T):
        shuffled_order = r.sample( range(1, len(train)+1), len(train))
        for index in shuffled_order:
            example = train[index]
            pred1 = VectorMult(ww1, example)
            if pred1 <= 1:
                ww1 = Update(ww1, curr_gamma, C[0], N, example)
            else:
                for j in range(len(ww1)):
                    ww1[j] = (1-curr_gamma) * ww1[j]

        shuffled_order = r.sample( range(1, len(train)+1), len(train))
        for index in shuffled_order:
            example = train[index]
            pred2 = VectorMult(ww2, example)
            if pred2 <= 1:
                ww2 = Update(ww2, curr_gamma, C[1], N, example)
            else:
                for j in range(len(ww2)):
                    ww2[j] = (1-curr_gamma) * ww2[j]

        shuffled_order = r.sample( range(1, len(train)+1), len(train))
        for index in shuffled_order:
            example = train[index]
            pred3 = VectorMult(ww3, example)
            if pred3 <= 1:
                ww3 = Update(ww3, curr_gamma, C[2], N, example)
            else:
                for j in range(len(ww3)):
                    ww3[j] = (1-curr_gamma) * ww3[j]

        t += 1 # Do this here? or after?
        curr_gamma = Schedule2(gamma, t)
        a = a * 0.5


    # With the weight vectors complete, we can now run tests with them
    error1 = 0
    error2 = 0
    error3 = 0  # For now we'll simply tally incorrect predictions on the data
    serr1 = 0
    serr2 = 0
    serr3 = 0
    for i in train:
        example = train[i]
        result = Test(w1, example)
        sign = CompSign(result)
        if sign != example.label:
            error1 += 1

        result = Test(w2, example)
        sign = CompSign(result)
        if sign != example.label:
            error2 += 1

        result = Test(w3, example)
        sign = CompSign(result)
        if sign != example.label:
            error3 += 1

        result = Test(ww1, example)
        if CompSign(result) != example.label:
            serr1 += 1

        result = Test(ww2, example)
        if CompSign(result) != example.label:
            serr2 += 1

        result = Test(ww3, example)
        if CompSign(result) != example.label:
            serr3 += 1

    # Error values using the first Schedule method
    error1 = error1 / len(train)
    error2 = error2 / len(train)
    error3 = error3 / len(train)

    # Error values for variants using the second Schedule method
    serr1 = serr1 / len(train)
    serr2 = serr2 / len(train)
    serr3 = serr3 / len(train)

    print("The error rates on the train data using Schedule 1 are: ")
    print("For C = 100/873: " + str(error1) + ", for C = 500/873: " + str(error2) + ", for C = 700/873: " + str(error2))

    print("The error rates on the train data using Schedule 2 are: ")
    print("For C = 100/873: " + str(serr1) + ", for C = 500/873: " + str(serr2) + ", for C = 700/873: " + str(serr3))

    print("")
    ## Need to add print statements and such later so that we can see the error rates between the different C's and data sets using Sched1

    terr1 = 0
    terr2 = 0
    terr3 = 0
    sterr1 = 0
    sterr2 = 0 
    sterr3 = 0
    for i in test:
        example = test[i]
        result = Test(w1, example)
        sign = CompSign(result)
        if sign != example.label:
            terr1 += 1

        result = Test(w2, example)
        sign = CompSign(result)
        if sign != example.label:
            terr2 += 1

        result = Test(w3, example)
        sign = CompSign(result)
        if sign != example.label:
            terr3 += 1

        result = Test(ww1, example)
        if CompSign(result) != example.label:
            sterr1 += 1

        result = Test(ww2, example)
        if CompSign(result) != example.label:
            sterr2 += 1

        result = Test(ww3, example)
        if CompSign(result) != example.label:
            sterr3 += 1

    terr1 = terr1 / len(test)
    terr2 = terr2 / len(test)
    terr3 = terr3 / len(test)

    sterr1 = sterr1 / len(test)
    sterr2 = sterr2 / len(test)
    sterr3 = sterr3 / len(test)

    ## Need to add print statements to do the same as above
    print("The error rates on the test data using Schedule 1 are: ")
    print("For C = 100/873: " + str(terr1) + ", for C = 500/873: " + str(terr2) + ", for C = 700/873: " + str(terr3))

    print("The error rates on the test data using Schedule 2 are: ")
    print("For C = 100/873: " + str(sterr1) + ", for C = 500/873: " + str(sterr2) + ", for C = 700/873: " + str(sterr3))

    print("")

    ######
    print("The parameters learned using C = 100/873 are as follows;")
    print("The parameters learned using Schedule 1 are: " + str(w1))
    print("The parameters learned using Schedule 2 are: " + str(ww1))
    print(" Difference in Training error is: " + str(error1-serr1) + ", and difference in test error is: " + str(terr1 - sterr1))

    print("The parameters learned using C = 500/873 are as follows;")
    print("The parameters learned using Schedule 1 are: " + str(w2))
    print("The parameters learned using Schedule 2 are: " + str(ww2))
    print(" Difference in Training error is: " + str(error2-serr2) + ", and difference in test error is: " + str(terr2 - sterr2))

    print("The parameters learned using C = 700/873 are as follows;")
    print("The parameters learned using Schedule 1 are: " + str(w3))
    print("The parameters learned using Schedule 2 are: " + str(ww3))
    print(" Difference in Training error is: " + str(error3-serr3) + ", and difference in test error is: " + str(terr3 - sterr3))


def DualDomain(train, test, T):
    w = [0,0,0,0,0]
    N = len(train)
    gamma = 1
    for i in range(T):
        shuffled_order = r.sample( range(1, len(train)+1), len(train))

    print("Todo")


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
            if float(terms[4]) == 0:
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
            if float(terms[4]) == 0:
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
