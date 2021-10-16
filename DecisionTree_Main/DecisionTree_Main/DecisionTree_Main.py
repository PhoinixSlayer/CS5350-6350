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
        self.Prediction = None  # If a leaf, the label that this path would predict for the example
        self.Value_of_Parent_Attribute = None  # The attribute that the parent node of this child split on
        self.Attribute_Split_With = ""  # The attribute that this specific node split on (if applicable)
        self.Value = ""  # Based on the attribute this nodes parent split on, this value refers to the subset of the parent's attribute group
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

# Fixed multiple version so that if any fractions are 0, 0 is added to the base total as log(0) isn't defined, its the solution 
#  mentioned in the slides and lectures for simplicity
def InfoGainMult(label_quantities, total_examples):
    total = 0
    for label in label_quantities:
        frac = label/total_examples
        if frac == 0
            total += 0
        else
            entr = -(frac)*m.log2(frac)
            total += entr
    return total


# Method that constructs the tree recursively
def SplitNode(sec_root, version):
    # --- Base Cases ---
    for l in sec_root.Labels_in_Branch.label_values_and_counts:
        if sec_root.Labels_in_Branch.label_values_and_counts[l] == sec_root.Labels_in_Branch.total_num_values:
            sec_root.Prediction = l
            return

    if len(sec_root.Remaining_Attributes) == 0:
        current_best = ""
        best = -1
        for l in sec_root.Labels_in_Branch.label_values_and_counts:
            if sec_root.Labels_in_Branch.label_values_and_counts[l] > best:
                best = sec_root.Labels_in_Branch.label_values_and_counts[l]
                current_best = l
        sec_root.Prediction = current_best
        return

    labels = []
    for l in sec_root.Labels_in_Branch.label_values_and_counts:
        labels.append[l]
    overall_gain = InfoGain(labels, sec_root.Labels_in_Branch.total_num_values)  # Here I need to calculate the total info gain using whatever method is desired.

    Atts_data = {}  # Data for each of the attributes we can work with
    for a in sec_root.Remaining_Attributes: # for this attribute ...
        temp = AttributeData(a)
        for v in car_attribute_values[a]: # for each of the attributes values ...
            temp.num_each_value.update({v: 0}) 
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
    gains = {}
    for a in sec_root.Remaining_Attributes:
        gains.update({a: 0})

    # Need to add option of calculation method here, so that each of the three IG, ME, and GI can be used depending on which is desired
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

    # Determine the best attribute to split on based on gathered data
    best_to_split = ""
    current = -1
    best = -1
    for a in gains:
        current = gains[a]
        if current > best:
            best = current
            best_to_split = a

    sec_root.Attribute_Split_With = a # Set attribute this node will split on to the one with the best gain
    attr_set = Atts_data[a]
    for v in attr_set.num_each_value:
        new = Node()
        new.Value_of_Parent_Attribute = a
        # new.Value = v  ## this is probably unneeded since we are accessing it with the att-value via the dictionary
        new.parent = sec_root
        new.Remaining_Attributes = sec_root.Remaining_Attributes[:]
        new.Remaining_Attributes.remove(a)
        new.Labels_in_Branch = sec_root.Labels_in_Branch
        new.Labels_in_Branch.total_num_values = 0
        for l in new.Labels_in_Branch.label_values_and_counts.values():
            l = 0
        # finish this part up by adding the node to the parents children list?
        sec_root.node_list.update({v: new})

    for example in sec_root.Examples_in_Branch.values():
        attribute = sec_root.Attribute_Split_With
        #ex_value = example.attributes[attribute]
        valued_node = sec_root.node_list[example.attributes[attribute]]
        valued_node.Examples_in_Branch.update({example.id: example})
        valued_node.Labels_in_Branch.total_num_values += 1
        valued_node.Labels_in_Branch.label_values_and_counts[example.label] += 1
        
    # With nodes created and added to the tree, all that is left is to run this work on those to further expand or complete the tree
    for n in sec_root.node_list.values():
        SplitNode(n, version)
    # I don't think there is any more work after this point, all that is needed now is adding ability for method to build tree to certain
    #  heights, make it so that the desired method of calculation is used based on the version variable, and that the tree works correctly
    # Later will need to figure out how to make it so the tree can take certain variables and crunch them into binary variables, but I think
    #  that that kind of work would happen in an earlier part of the algorithm.
    # I think I am probably missing a few things, but I won't be able to make sure until later after I've rested a bit and can look over
    #  everything with a fresh start, ready to (mostly) finish this part and hopefully start moving on by preparing something for HW2



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