# File for main project, Hoping this works and fixes my problem
#import the-things-I-need
import pandas as pd

class Example:
    #example_number = -1
    #attributes     = []
    #label          = ""
    def __init__(self, ex_number, attributes, lbl):
        self.example_number = ex_number
        self.attributes = attributes
        self.label = lbl

class Attribute:
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


#Beginning of file that does the work I need
def main():
    car_labels = Label_data(0, { "unacc": 0, "acc": 0, "good": 0, "vgood": 0 })

    car_attributes  = [ ]
    bank_attributes = [ ]

    # Load and process the car data into data storage structures and process them into a complete tree (as far as it's depth lets it go)
    # We'll look at depth specific data later.
    id = 1
    with open ( "car/train.csv" , 'r' ) as f:
        for l in f:
            terms = l.strip().split(',')
            # ... process example
            id += 1
            if id % 100 == 0:
                print(terms)
    
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




main()