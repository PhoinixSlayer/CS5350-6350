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
def UpdateWeight(w, e, rate):
    for i in range( len(e.attributes) ):
        e.attributes[i] = e.attributes[i] * e.label * rate
    new_weight = []
    for i in range( len(w) ):
        new_weight.append( (w[i] + e.attributes[i]) )
    return new_weight

##
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


##
def StandardPerceptron(training, test_examples, T):
    epoch = []
    weight = [0,0,0,0,0] # not sure if I should have one for each epoch or just one total throughout the whole process
    rate = 1
    R = m.sqrt( (1 + len(training)) )
    
    for i in range(T):
        shuffled_order = r.sample( range(1, len(training)+1), len(training))
        for ex in shuffled_order:
            example = training[ex]
            result = VectorMult(weight, example)
            if result <= 0:
                weight = UpdateWeight(weight, example, rate)
                rate = rate * 0.9

    # After process completes, we have the learned weight vector, all we need to do now is use it to predict each test example and report
    #  the average error for the learned weight vector.
    correct = 0
    incorrect = 0
    for ex in test_examples:
        example = test_examples[ex]
        prediction = Predict(weight, example)
        if prediction > 0:
            if example.label == 1:
                correct += 1
            else:
                incorrect += 1
        else:
            if example.label == 0:
                correct += 1
            else:
                incorrect += 1

    error_average = incorrect / (correct + incorrect)

    print("The learned weight vector for Standard Perceptron is: ", weight)
    print("And the average error for this weight when used to predict the test data is: " + str(error_average))




def Compare(curr, next):
    for i in range(len(curr)):
        if curr[i] == next[i]:
            continue
        else:
            return False
    return True


def DetermineUniques(stages, counts, unique_weights, unique_counts):
    stage_used = []
    for i in range(len(stages)):
        stage_used.append(False)
    ii = 0
    for i in range(len(stages)):
        if stage_used[i] == False:
            unique_weights.append(stages[i])
            unique_counts.append(counts[i])
            stage_used[i] = True

            current = stages[i]
            if i + 1 < len(stages):
                for j in range(i+1, len(stages)):
                    next_w = stages[j]
                    if Compare(current, next_w):
                        unique_counts[ii] += counts[j]
                        stage_used[j] = True
            ii += 1


###
def VotedPerceptron(training, test, T):
    stage_weight = []
    stage_count = []
    mm = 0
    weight = [0,0,0,0,0] # not sure if I should have one for each epoch or just one total throughout the whole process
    rate = 1
    R = m.sqrt( (1 + len(training)) )

    stage_weight.append(weight)
    stage_count.append(0)
    
    for i in range(T):
        shuffled_order = r.sample( range(1, len(training)+1), len(training))
        for ex in shuffled_order:
            example = training[ex]
            result = VectorMult(weight, example)
            if result <= 0:
                weight = UpdateWeight(weight, example, rate)
                mm += 1

                stage_weight.append(weight)
                stage_count.append(0)

                stage_count[mm] += 1

                rate = rate * 0.9
            else:
                stage_count[mm] += 1

    # Find the unique weight vectors here
    unique_weights = []
    unique_counts = []
    DetermineUniques(stage_weight, stage_count, unique_weights, unique_counts)


    # Do similar calculations like before, determining if it correctly predicts a test examples label
    correct = 0
    incorrect = 0
    for ex in test:
        example = test[ex]
        total = 0
        for i in range(len(stage_weight)):
            w = stage_weight[i]
            c = stage_count[i]
            pred = Predict(w, example)
            sign = 0
            if pred >= 0:
                sign = 1
            else:
                sign = -1
            total += c * sign
        # After summing all weights, the sign of the total is the prediction for the test example
        final_prediction = 0
        if total > 0:
            final_prediction = 1
        else:
            final_prediction = 0

        # With the sign of the total prediction complete, we can then determine if, on average, the voted perceptron correctly predicted 
        #  this examples label.
        if final_prediction == example.label:
            correct += 1
        else:
            incorrect += 1

    error = incorrect / (incorrect + correct)

    print("The distinct weight vectors and their respective counts in Voted Perceptron are as follows.")
    print("The number of total weights is: " + str(len(stage_weight)))
    print("The number of unique weights is: " + str(len(unique_weights)))
    for i in range(len(unique_weights)):
        print(unique_weights[i], ", count: " + str(unique_counts[i]))
    print("The average prediction error for Voted Perceptron is: " + str(error))
    # the above value uses all vectors, not just the unique ones

    

###
def AveragePerceptron(training, test, T):
    stage_weight = []
    stage_count = []
    mm = 0
    weight = [0,0,0,0,0] # not sure if I should have one for each epoch or just one total throughout the whole process
    averaged_weight = [0,0,0,0,0]
    rate = 1
    R = m.sqrt( (1 + len(training)) )

    stage_weight.append(weight)
    stage_count.append(0)
    
    for i in range(T):
        shuffled_order = r.sample( range(1, len(training)+1), len(training))
        for ex in shuffled_order:
            example = training[ex]
            result = VectorMult(weight, example)
            if result <= 0:
                weight = UpdateWeight(weight, example, rate)
                mm += 1

                stage_weight.append(weight)
                stage_count.append(0)

                stage_count[mm] += 1

                rate = rate * 0.9
            else:
                stage_count[mm] += 1

            for i in range(len(averaged_weight)):
                averaged_weight[i] += weight[i]

    correct = 0
    incorrect = 0
    for ex in test:
        example = test[ex]
        total = 0
        pred = Predict(averaged_weight, example)
        final_prediction = 0
        if pred > 0:
            final_prediction = 1
        else:
            final_prediction = 0

        # With the sign of the total prediction complete, we can then determine if, on average, the voted perceptron correctly predicted 
        #  this examples label.
        if final_prediction == example.label:
            correct += 1
        else:
            incorrect += 1

    error = incorrect / (incorrect + correct)
    
    print("The learned weight vector for Average Perceptron is: ", averaged_weight)
    print("The average prediction error for Average Perceptron is: " + str(error))
    #print("The total number of learned weight vectors is (including the initial w=0 vector): " + str(len(stage_weight)))



def main():
    train_examples = {}
    test_examples = {}
    
    id = 1
    with open ( "bank-note/train.csv" , 'r' ) as f: 
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3]) ]
            temp = Example(id, attributes, float(terms[4]))  # Label of example, if 0 means forged, 1 means genuine
            train_examples.update({id: temp})
            id += 1
            
    id = 1
    with open( "bank-note/test.csv", 'r') as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3]) ]
            temp = Example(id, attributes, float(terms[4]))
            test_examples.update({id: temp})
            id += 1

    ### Perceptron ###
    T = 10

    StandardPerceptron(train_examples, test_examples, T)

    VotedPerceptron(train_examples, test_examples, T)

    AveragePerceptron(train_examples, test_examples, T)
    
    ### Not sure how I'll do it yet, but need to gather the average prediction error for each of the three and then compare them.
    


### Run Program ###
main()