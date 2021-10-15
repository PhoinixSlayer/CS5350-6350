# File for main DecisionTree project
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
        self.value_label_counts = { }   #make into a default dictionary, each key is attribute value, goes to array 0 <-> labels-1
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
        self.Value_of_Parent_Attribute = None  # The value of the attribute this node was created with
        self.Attribute = None  # Attribute this Node is splitting on
        self.Examples_in_Branch = {}  # The examples that we can use in this section of the tree
        self.Labels_in_Branch = None  # The label data for the above examples
        self.Remaining_Attributes = []  # The attributes that we haven't used in this section of the tree
        self.node_list = {}  
        self.parent = None  # This nodes parent node



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
    for label in label_quantities:
        frac = label/total_examples
        entr = -(frac)*m.log2(frac)
        total += entr
    return total


# Method that constructs the tree recursively
def SplitNode(sec_root, version):
    # The base cases would go here before any of the math occurs, as much of it depends on several pieces of information being non-zero
    # --- Base Cases ---

    labels = []
    for l in sec_root.Labels_in_Branch.label_values_and_counts:
        labels.append[l]
    overall_gain = InfoGain(labels, sec_root.Labels_in_Branch.total_num_values)  # Here I need to calculate the total info gain using whatever method is desired.

    Atts_data = {}  # Data for each of the attributes we can work with
    for a in sec_root.Remaining_Attributes: # for this attribute ...
        temp = AttributeData(a)
        #Atts_data.update({a:temp})
        for v in car_attribute_values[a]: # for each of the attributes values ...
            temp.num_each_value.update({v: 0}) # 
            temp.value_label_counts.update({v: {}})
            for l in sec_root.Labels_in_Branch.label_values_and_counts:
                temp.value_label_counts[v].update({l: 0})
        Atts_data.update({a:temp})
    
    # with both of the dictionaries prepared, we can now fill them with the appropriate attribute data
    for e in sec_root.Examples_in_Branch:
        example = sec_root.Examples_in_Branch[e]
        for a in sec_root.Remaining_Attributes:
            data = Atts_data[a]
            value = example.attributes[a]
            data.num_each_value[value] += 1
            data.value_label_counts[value][example.label] += 1
    # After this point, all of the data should be accurately totaled so we can calculate the desired info

    # After each of the attributes have been built and the data correctly parsed, we can then do our calculations to determine which one is
    #  best to split on at this stage of the tree
    # First we need to figure out what the gains for each of the attributes values are, then we can combine those to calculate the total gain
    #  for the attribute. after we've done that for everything, we can decide which is the best, and use that to split this node on the tree.
    gains = {}
    for a in sec_root.Remaining_Attributes:
        gains.update({a: 0})

    for a in sec_root.Remaining_Attributes:
        value_total = 0
        current_attr = Atts_data[a]
        for v in current_attr.num_each_value:
            for l in current_attr.value_label_counts[v].values():
                label_total_list.append(l)
            total_for_value = current_attr.num_each_value[v]
            value_sum = InfoGainMult(label_total, total_for_value) # Calculate this attr value's entropy
            value_total += value_sum * (total_for_value / sec_root.Labels_in_Branch.total_num_values) # mult it by the proportion of the data it holds
        # Pretty sure that with each of the values of the attribute determined, its just the overall gain minus this total
        gains[a] = overall_gain - value_total

    # Here, its a simple check to determine which attribute has the highest score (and will be the one that we split on)
    best_to_split = ""
    current = -1
    best = -1
    for a in gains:
        current = gains[a]
        if current > best:
            best = current
            best_to_split = a

    #TODO: with splitter decided, need to create node for each attribute subset, accurately distribute data, set up tree connections, 
    #       fill in base cases so errors don't happen and tree properly completes, add different tree level functionality, (+ more)



# Side note: calculation_version refers to which method of information gain to use. 
#  If 0, use Entropy (or Info Gain); if 1, use ME; if 2, use Gini Index
def DetermineTree(all_examples, total_labels, calculation_version, attributes_to_use):
    print("Not Done")
    root = Node()
    root.Labels_in_Branch = total_labels
    root.Examples_in_Branch = all_examples
    root.Remaining_Attributes = attributes_to_use # I think this makes a copy of the original, so when this is edited the other stays unchanged

    SplitNode(root, calculation_version)



car_attribute_values = { "buying": {"vhigh", "high", "med", "low"},
                       "maint": {"vhigh", "high", "med", "low"},
                      "doors": {"2", "3", "4", "5more"},
                     "persons": {"2", "4", "more"},
                    "lug_boot": {"small", "med", "big"},
                   "safety": {"low", "med", "high"} }

def main():
    car_labels = Label_data(0, { "unacc": 0, "acc": 0, "good": 0, "vgood": 0 })

    #car_attributes_data  = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    #bank_attributes = [ ]

    train_examples = {}
    test_examples = {}
    # Load and process the car data into data storage structures and process them into a complete tree (as far as it's depth lets it go)
    # We'll look at depth specific data later.
    id = 1
    with open ( "car/train.csv" , 'r' ) as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = {"buying":terms[0], "maint":terms[1], "doors": terms[2], "persons": terms[3], "lug_boot": terms[4], "safety": terms[5]}
            temp = Example(id, attributes, terms[6])
            train_examples.update({id: temp})

            car_labels.total_num_values += 1
            car_labels.label_values_and_counts[temp.label] += 1
            id += 1
            #if id % 100 == 0:
             # print(terms)
    
    # Upon completion of data extraction, calculate a decision tree for each method of info gain using the data
    #  We can add tree length as a factor later, right now I just want to get my code to make a tree without huge problems
    DetermineTree(train_examples, car_labels, 0, ["buying", "maint", "doors", "persons", "lug_boot", "safety"])
    DetermineTree(train_examples, car_labels, 1, ["buying", "maint", "doors", "persons", "lug_boot", "safety"])
    DetermineTree(train_examples, car_labels, 2, ["buying", "maint", "doors", "persons", "lug_boot", "safety"])
    


    
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