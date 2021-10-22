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
    return (cost_ex * xi)

# Returns total gradient for the weight vector as a new vector.
def Gradient(curr_weight, examples):
    gradient = [0,0,0,0,0,0,0,0]
    for wi in range(8):
        wi_sum = 0
        for ex in examples:
            example = examples[ex]
            xi = example.attributes[wi]
            cost_ex = GradientPart(curr_weight, example, xi)
            wi_sum += cost_ex
        gradient[wi] = -wi_sum
    return gradient


## 
def BatchGradientDecent(train_examples, test_examples):
    losses_on_training = [] # Need to keep track of the total loss for every weight w_i, because they want me to make a figure with the
    stage_iteration = []
    #  weight costs to show how it changed with each update
    final_vector = []  # At the end they want to run the test data with the final weight vector to determine the loss (total cost) for the
    #  final weight vector using the test data.
    norm_difference = 1
    current_weight = [0, 0, 0, 0, 0, 0, 0, 0]
    current_rate = 1
    prev_rate = current_rate
    stage = 0
    
    while norm_difference > 10**(-6):
        curr_loss = Loss(current_weight, train_examples)
        losses_on_training.append(curr_loss)
        stage_iteration.append(stage)
        # We have the previous Cost, the current weight vector, and the current rate, need to calculate next weight vector
        gradient_of_current = Gradient(current_weight, train_examples)
        new_weight = []
        for gi in range(len(gradient_of_current)):
            gradient_of_current[gi] = gradient_of_current[gi] * current_rate
        for i in range(len(gradient_of_current)):
            new_weight.append(current_weight[i] - gradient_of_current[i])
        prev_rate = current_rate
        current_rate = current_rate / 2

        sum = 0
        for i in range(len(current_weight)):
            sum += new_weight[i] - current_weight[i]
        norm_difference = sum
        current_weight = new_weight
        stage = 1

    # When the above stops because the convergence is low enough, use the current weight vector (because it was set to the new - and
    #  thus final - weight vector as that was the one that brought the convergence low enough.
    print("The final weight vector is: ", current_weight)
    print("And the learning rate that was previously used to create it is: " + str(prev_rate))

    # Now use the stored total costs (or loss) of the training data at each step
    pyp.plot(stage_iteration, losses_for_training, label = "Cost for Training Data at each stage")
    pyp.xlabel("Current stage")
    pyp.ylabel("Loss on Training Set")
    pyp.title("Depiction of the Loss for the Weight Vector on the Training Set")
    pyp.legend()
    pyp.show()

    # -- We don't have the final weight vector's loss yet, so calculate it here --
    test_loss = Loss(current_weight, test_examples)
    print("The total cost for the Test set using the final weight vector is: " + str(test_loss))


## 
def StochasticGradientDecent(train_examples, test_examples):
    current_weight = [0,0,0,0,0,0,0,0]
    current_rate = 1

    rand_order = select(range(1, len(train_examples)), len(train_examples))

    for e in rand_order:
        example = train_examples[e]


    print("Todo")


##
def OptimalCalculation(train_examples, test_examples):
    print("Todo")



def main():
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

    BatchGradientDecent(train_examples, test_examples)

    StochasticGradientDecent(train_examples, test_examples)

    OptimalCalculation(train_examples, test_examples)



# Run program
main()