# File for main project, Hoping this works and fixes my problem
#import the-things-I-need
import math as m
import pandas as pd

class Example:
    #example_number = -1
    #attributes     = []
    #label          = ""
    def __init__(self, ex_number, attributes, lbl):
        self.example_number = ex_number
        self.attributes = attributes
        self.label = lbl

class AttributeData:
    #attribute_name          = ""
    #num_each_value          = {}
    #attribute_value_counts  = {}
    #attribute_values_neg
    def __init__(self, attr_name):
        self.attribute_name = attr_name          
        self.num_each_value = { }
        self.attribute_value_counts = { }   #make into a default dictionary, each key is attribute value, goes to array 0 <-> labels-1
     #   self.attribute_values_neg = { }   #make into a default dictionary (correct?)

    
class Label_data:
    #total_num_labels        = 0 # equivalent to saying total number of examples
    #label_values_and_counts = {}
    def __init__(self, total_num, labels_and_values):
        self.total_num_values = total_num
        self.label_values_and_counts = labels_and_values


# Class for the decision tree representation of the data
class Node:
    def __init__(self):
        self.Attribute_of_piece = None
        self.node_list = []
        self.parent = None



## Each of these are binary versions, where there are only two possibilities for the attribute
# Can't quite remember, need to check
def InfoGain(positive_examples, negative_examples, total_examples):
    pos = positive_examples/total_examples
    neg = negative_examples/total_examples
    return -(pos*m.log2(pos))-(neg*m.log2(neg))

# Pretty sure this is correct, but need to double check
def MajorityError(majority_examples, total_examples):
    return 1 - majority_examples/total_examples
    
# Pretty sure this is correct, but need verification
def GiniIndex(positive_examples, negative_examples, total_examples):
    pos = positive_examples/total_examples
    neg = negative_examples/total_examples
    return 1 - (pos**2 + neg**2)

## These are the versions of the above that calculate the info gain for a given attribute based on multiple label values.
# Based on my foggy memory, this is what the entropy version should look like for some number of possible labels. The idea is
#  that if there are multiple labels an example can have, the total entropy for an attribute value is the combination of each of the 
#  different label quantities each divided by the total number of examples in this attribute value.
def GiniIndexMult(label_quantities, total_examples):
    total = 0.0
    for labels in label_quantities:
        frac = labels/total_examples
        total += frac**2
    return 1 - total
#Don't know why, but for some reason something's wrong with GiniIndexMult?

def InfoGainMult(label_quantities, total_examples):
    total = 0
    for labels in label_quantities:
        frac = labels/total_examples
        entr = -((frac)*m.log2(frac))
        total += entr
    return total



# Side note: calculation_version refers to which method of information gain to use. 
#  If 0, use Entropy (or Info Gain); if 1, use ME; if 2, use Gini Index
def DetermineTree(all_examples, total_labels, calculation_version):
    print("Not Done")
    r = InfoGain(4, 2, 6)
    s = GiniIndexMult([1, 1], 2)


#Beginning of file that does the work I need
def main():
    car_labels = Label_data(0, { "unacc": 0, "acc": 0, "good": 0, "vgood": 0 })

    #car_attributes_data  = [ ]
    #bank_attributes = [ ]

    train_examples = {}
    test_examples = {}
    # Load and process the car data into data storage structures and process them into a complete tree (as far as it's depth lets it go)
    # We'll look at depth specific data later.
    id = 1
    with open ( "car/train.csv" , 'r' ) as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = [terms[0], terms[1], terms[2], terms[3], terms[4], terms[5]]
            temp = Example(id, attributes, terms[6])
            train_examples.update({id: temp})

            car_labels.total_num_values += 1
            car_labels.label_values_and_counts[temp.label] += 1
            id += 1
            #if id % 100 == 0:
             # print(terms)
    
    # Upon completion of data extraction, calculate a decision tree for each method of info gain using the data
    #  We can add tree length as a factor later, right now I just want to get my code to make a tree without huge problems
    DetermineTree(train_examples, car_labels, 0)
    DetermineTree(train_examples, car_labels, 1)
    DetermineTree(train_examples, car_labels, 2)
    


    
    id = 1
    with open( "car/test.csv", 'r') as f:
        for l in f:
            terms = l.strip().split(',')
            # process test example
            id += 1





    # Deal with later, make sure the car data looks good when processed.

    # Process bank data and create whole tree with it
    bank_label = Label_data(0, { "yes": 0, "no": 0 })
    id = 1
    with open( "bank/train.csv", 'r' ) as f:
        for l in f:
            terms = l.strip().split(',')
            id += 1

    id = 1
    with open( "bank/test.csv", 'r' ) as f:
        for l in f:
            terms = l.strip().split(',')
            id += 1

    


# Execute program
main()