# Main file for Ensemble Learning homework portion.

import math as m
import pandas as pd

class Example:
    def __init__(self, ex_number, attributes, lbl):
        self.example_number = ex_number
        self.attributes = attributes
        self.label = lbl

class AttributeData:
    def __init__(self, attr_name):
        self.attribute_name = attr_name
        self.num_each_value = { }
        self.value_label_counts = { }

class Label_data:
    def __init__(self, total_num, labels_and_values):
        self.total_num_values = total_num  # Represents number of examples tangentially
        self.label_values_and_counts = labels_and_values  # Represents the totals for each label in a dictionary

# Class for the decision tree representation of the data
class Node:
    def __init__(self):
        self.Prediction = None  # If a leaf, the label that this path would predict for the example
        self.Value_of_Parent_Attribute = None  # The attribute that the parent node of this child split on
        self.Attribute_Split_With = ""  # The attribute that this specific node split on (if applicable)
        self.Examples_in_Branch = {}  # The examples that we can use in this section of the tree
        self.Labels_in_Branch = None  # The label data for the above examples
        self.Remaining_Attributes = []  # The attributes that we haven't used in this section of the tree
        self.node_list = {}
        self.parent = None  # This nodes parent node



# Method for calculating gain when labels are binary
def InfoGain(positive_examples, negative_examples, total_examples):
    pos = positive_examples/total_examples
    neg = negative_examples/total_examples
    return -(pos*m.log2(pos))-(neg*m.log2(neg))

def MajorityError(majority_examples, total_examples):
    # Some att-values with have no examples to their subgroup, which means in the gains calculation its ME=0 times the proportion of this value
    #  among the total examples in that subset
    if total_examples == 0:
        return 0
    return 1 - majority_examples/total_examples
    
# Method for calculating GI when possible labels are binary
def GiniIndex(positive_examples, negative_examples, total_examples):
    pos = positive_examples/total_examples
    neg = negative_examples/total_examples
    return 1 - (pos**2 + neg**2)

## These are the versions of the above that calculate the info gain for a given attribute based on multiple label values.
def GiniIndexMult(label_quantities, total_examples):
    total = 0.0
    for labels in label_quantities:
        frac = labels/total_examples
        total += frac**2
    return 1 - total

# Fixed multiple version so that if any fractions are 0, 0 is added to the base total as log(0) isn't defined, its the solution 
#  mentioned in the slides and lectures for simplicity
def InfoGainMult(label_quantities, total_examples):
    total = 0
    for label in label_quantities:
        frac = label/total_examples
        if frac == 0:
            total += 0
        else:
            entr = -(frac)*m.log(frac,len(label_quantities))
            total += entr
    return total



## Method that constructs the tree recursively
# Doesn't make the most accurate tree as of right now, but it's at least something to start with
def SplitNode(sec_root, version, current_level, desired_level):
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
    # Check on the depth of the tree would go here to see if we've reached the desired length
    if current_level == desired_level:
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
        labels.append(sec_root.Labels_in_Branch.label_values_and_counts[l])
    if version == 0: # Here we use the standard Information Gain
        overall_gain = InfoGainMult(labels, sec_root.Labels_in_Branch.total_num_values)
    if version == 1: # Here we use Majority Error
        best = -1;
        for lab in labels:
            if lab > best:
                best = lab
        overall_gain = MajorityError(best, sec_root.Labels_in_Branch.total_num_values)
    if version == 2: # Here we use the Gini Index
        overall_gain = GiniIndexMult(labels, sec_root.Labels_in_Branch.total_num_values)

    Atts_data = {}
    for a in sec_root.Remaining_Attributes: # for this attribute ...
        temp = AttributeData(a)
        for v in bank_attribute_values[a]: # for each of the attributes values ...
            temp.num_each_value.update({v: 0}) 
            temp.value_label_counts.update({v: {}})
            for l in sec_root.Labels_in_Branch.label_values_and_counts:
                temp.value_label_counts[v].update({l: 0})
        Atts_data.update({a:temp})
    
    # With both of the dictionaries prepared, we can now fill them with the appropriate attribute data
    for e in sec_root.Examples_in_Branch:
        example = sec_root.Examples_in_Branch[e]
        for a in sec_root.Remaining_Attributes:
            data = Atts_data[a]
            value = example.attributes[a]
            data.num_each_value[value] += 1
            data.value_label_counts[value][example.label] += 1

    # After each of the attributes have been built and the data correctly parsed, we can then do our calculations to determine which one is
    #  best to split on at this stage of the tree
    gains = {}
    for a in sec_root.Remaining_Attributes:
        gains.update({a: 0})

    for a in sec_root.Remaining_Attributes:
        value_total = 0
        current_attr = Atts_data[a]
        for v in current_attr.num_each_value:
            label_total_list = []
            for l in current_attr.value_label_counts[v].values():
                label_total_list.append(l)
            total_for_value = current_attr.num_each_value[v]
            if version == 0:
                if total_for_value == 0:
                    value_total += 0
                else:
                    value_sum = InfoGainMult(label_total_list, total_for_value) # Calculate this attr value's entropy
                    value_total += value_sum * (total_for_value / sec_root.Labels_in_Branch.total_num_values) # mult it by the proportion of the data it holds
            if version == 1:
                best = -1
                for label in label_total_list:
                    if label > best:
                        best = label
                value_sum = MajorityError(best, total_for_value)
                value_total += value_sum * (total_for_value / sec_root.Labels_in_Branch.total_num_values)
            if version == 2:
                if total_for_value == 0:
                    value_total += 0
                else:
                    value_sum = GiniIndexMult(label_total_list, total_for_value)
                    value_total += value_sum * (total_for_value / sec_root.Labels_in_Branch.total_num_values)
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
        new.parent = sec_root
        new.Remaining_Attributes = sec_root.Remaining_Attributes[:]
        new.Remaining_Attributes.remove(a)
        new.Labels_in_Branch = Label_data(0, {})
        new.Labels_in_Branch.label_values_and_counts = sec_root.Labels_in_Branch.label_values_and_counts.copy()
        new.Labels_in_Branch.total_num_values = 0  # same thing here as below
        for l in new.Labels_in_Branch.label_values_and_counts:
            new.Labels_in_Branch.label_values_and_counts[l] = 0  # change so we're reusing the data collected in the attribute set, just transfer the label values for this value-node
        sec_root.node_list.update({v: new})
        
    for example in sec_root.Examples_in_Branch.values():
        valued_node = sec_root.node_list[example.attributes[a]]
        valued_node.Examples_in_Branch.update({example.example_number: example})
        valued_node.Labels_in_Branch.total_num_values += 1
        valued_node.Labels_in_Branch.label_values_and_counts[example.label] += 1
        
    # With nodes created and added to the tree, all that is left is to run this work on those to further expand or complete the tree
    for n in sec_root.node_list.values():
        SplitNode(n, version, current_level+1, desired_level)
    # Later will need to figure out how to make it so the tree can take certain variables and crunch them into binary variables, but I think
    #  that that kind of work would happen in an earlier part of the algorithm.



# Side note: calculation_version refers to which method of information gain to use. 
#  If 0, use Entropy (or Info Gain); if 1, use ME; if 2, use Gini Index
def DetermineTree(all_examples, total_labels, calculation_version, desired_level, attributes_to_use):
    root = Node()
    root.Labels_in_Branch = total_labels
    root.Examples_in_Branch = all_examples
    root.Remaining_Attributes = attributes_to_use # I think this makes a copy of the original, so when this is edited the other stays unchanged

    SplitNode(root, calculation_version, 0, desired_level)
    return root



## Method used to test tree on all examples and gather data for accuracy comparisons
def TestTree(tree_root, examples_to_test):
    example_predictions = {}
    for e in examples_to_test:
        example = examples_to_test[e]
        current_node = tree_root
        prediction_from_tree = None
        while prediction_from_tree == None:
            attr_to_follow = current_node.Attribute_Split_With
            if attr_to_follow == "":
                prediction_from_tree = current_node.Prediction
            else:
                ex_value = example.attributes[attr_to_follow]
                current_node = current_node.node_list[ex_value]
        example_predictions.update({example.example_number: prediction_from_tree})

    # After deriving the prediction for each example, iterate through them to find the number of hits and misses
    correct_predictions = 0
    wrong_predictions = 0
    for id in example_predictions:
        prediction = example_predictions[id]
        orig_example_label = examples_to_test[id].label
        if prediction == orig_example_label:
            correct_predictions += 1
        else:
            wrong_predictions += 1

    accuracy = correct_predictions / len(examples_to_test)

    print("The accuracy for this tree is: " + str(accuracy))


# Method that converts numerical attributes to binary attributes. Here, we're simply going to treat the two values as over the median or under,
#  similar to how the binary label for the Kaggle project is over (or equal to) 50k or under 50k a year
def TransformNumericals(examples, test_set):
    for att in attributes_to_convert:
        vals_as_int = []
        for ex in examples:
            vals_as_int.append( int(examples[ex].attributes[att]) )
        # Sort all of the numerical values in the data of this attribute so that its smallest->biggest
        vals_as_int.sort()
        median_val = vals_as_int[int(len(vals_as_int)/2)]
        for ex in examples:
            string_val = examples[ex].attributes[att]
            if int(string_val) >= median_val:
                examples[ex].attributes[att] = "over"
            else:
                examples[ex].attributes[att] = "under"
        for ex in test_set:
            string_val = test_set[ex].attributes[att]
            if int(string_val) >= median_val:
                test_set[ex].attributes[att] = "over"
            else:
                test_set[ex].attributes[att] = "under"


# All attributes currently empty are the ones that need to be converted to binary attributes
bank_attribute_values = {"age": {"over", "under"}, 
                         "job": {"admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"}, 
                         "marital": {"married","divorced","single"}, 
                         "education": {"unknown","secondary","primary","tertiary"}, 
                         "default": {"yes","no"}, 
                         "balance": {"over", "under"}, 
                         "housing": {"yes","no"},
                         "loan": {"yes","no"}, 
                         "contact": {"unknown","telephone","cellular"}, 
                         "day": {"over", "under"}, 
                         "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}, 
                         "duration": {"over", "under"}, 
                         "campaign": {"over", "under"}, 
                         "pdays": {"over", "under"},
                         "previous": {"over", "under"}, 
                         "poutcome": {"unknown","other","failure","success"}}

attributes_to_convert = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

def main():
    bank_train_labels = Label_data(0, {"yes": 0, "no": 0})
    bank_test_labels = Label_data(0, {"yes": 0, "no": 0})

    train_examples = {}
    test_examples = {}
    
    id = 1
    with open ( "bank/train.csv" , 'r' ) as f: 
        for l in f:
            terms = l.strip().split(',')
            attributes = {"age": terms[0], "job": terms[1], "marital": terms[2], "education": terms[3], "default": terms[4], 
                          "balance": terms[5], "housing": terms[6], "loan": terms[7], "contact": terms[8], "day": terms[9], 
                          "month": terms[10], "duration": terms[11], "campaign": terms[12], "pdays": terms[13], "previous": terms[14], 
                          "poutcome": terms[15]}
            temp = Example(id, attributes, terms[16]) 
            train_examples.update({id: temp})

            bank_train_labels.total_num_values += 1
            bank_train_labels.label_values_and_counts[temp.label] += 1
            id += 1
            
    id = 1
    with open( "bank/test.csv", 'r') as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = {"age": terms[0], "job": terms[1], "marital": terms[2], "education": terms[3], "default": terms[4], 
                          "balance": terms[5], "housing": terms[6], "loan": terms[7], "contact": terms[8], "day": terms[9], 
                          "month": terms[10], "duration": terms[11], "campaign": terms[12], "pdays": terms[13], "previous": terms[14], 
                          "poutcome": terms[15]}
            temp = Example(id, attributes, terms[16])
            test_examples.update({id: temp})

            bank_test_labels.total_num_values += 1
            bank_test_labels.label_values_and_counts[temp.label] += 1
            id += 1

    TransformNumericals(train_examples, test_examples)
    # After this point the numerical attributes should be properly represented as binary attributes based around the median in the data.

    # In this instance of the homework, I don't think tree depth is ever going to be beyond 1, so this won't be necessary here.
    #  I just need the tree method I've built to work with the bank data properly
    requested_level = 0;

    ## Based on the assignment description we're only using the default entropy version, so for now all other versions won't be dealt with
    ## Additionally, normally we'd construct trees of size 1-16, but becuase they want stumps we'll only be using 1 for the depth limit
    GainRoot = DetermineTree(train_examples, bank_train_labels, 0, 1, ["age", "job", "marital", "education", "default", "balance", "housing",
                                                                 "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                                                                 "previous", "poutcome"])
    # The attributes I will need to convert to binary using the mean conversion are: age, balance, day, duration, campaign, pdays, previous.

    ## Won't be doing testing or anything yet, need the program to compress numerical and continuous data properly before even making
    ##  basic trees and getting the new work done.
    TestTree(GainRoot, train_examples)

    ## Different work is being done, so I don't think I'll be using all of these, especially not yet
    TestTree(GainRoot, test_examples)


    ### AdaBoost stuff will go here after this section header


    ### Bagging implementation goes here, going to work on this first because it seems like it needs the least modification


    


# Execute program
main()