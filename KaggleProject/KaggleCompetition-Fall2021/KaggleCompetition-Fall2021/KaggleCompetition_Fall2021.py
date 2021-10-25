# Main file for Ensemble Learning homework portion.

import csv

import random as r
import math as m
import pandas as pd
# libraries for ease of convenience, used to create graphs with the collected data
import matplotlib as mat
import matplotlib.pyplot as pyp

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


# Method used to select random attributes from a total selection base
def PickRandomFrom(available_attributes):
    # This is a very hackjob way of going about it, it should work, but it will be very bad performance wise.
    used_attr = {}
    attr_subset = []
    if len(available_attributes) <= RANDOM_LIMIT:
        attr_subset = available_attributes[:]
        return attr_subset
    random_set = r.sample(range(0, len(available_attributes)-1), RANDOM_LIMIT)
    for i in random_set:
        attr_subset.append(available_attributes[i])
    return attr_subset


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

    # Compute the overall gain for this section of the tree
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

    # Assemble data structures for the attributes that we can use at this stage for later calculations to determine best split.
    if DOING_FORESTS:
        random_attributes = PickRandomFrom(sec_root.Remaining_Attributes)
        Atts_data = {}
        for a in random_attributes: # for this attribute ...
            temp = AttributeData(a)
            for v in bank_attribute_values[a]: # for each of the attributes values ...
                temp.num_each_value.update({v: 0}) 
                temp.value_label_counts.update({v: {}})
                for l in sec_root.Labels_in_Branch.label_values_and_counts:
                    temp.value_label_counts[v].update({l: 0})
            Atts_data.update({a:temp})
    else:
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
        for a in Atts_data:
            data = Atts_data[a]
            value = example.attributes[a]
            data.num_each_value[value] += 1
            data.value_label_counts[value][example.label] += 1

    # After each of the attributes have been built and the data correctly parsed, we can then do our calculations to determine which one is
    #  best to split on at this stage of the tree
    gains = {}
    for a in Atts_data:
        gains.update({a: 0})

    for a in Atts_data:
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
        new.Labels_in_Branch.total_num_values = 0
        for l in new.Labels_in_Branch.label_values_and_counts:
            new.Labels_in_Branch.label_values_and_counts[l] = 0
        sec_root.node_list.update({v: new})
        
    for example in sec_root.Examples_in_Branch.values():
        valued_node = sec_root.node_list[example.attributes[a]]
        valued_node.Examples_in_Branch.update({example.example_number: example})
        valued_node.Labels_in_Branch.total_num_values += 1
        valued_node.Labels_in_Branch.label_values_and_counts[example.label] += 1

    # With nodes created and added to the tree, all that is left is to run this work on those to further expand or complete the tree
    for n in sec_root.node_list.values():
        SplitNode(n, version, current_level+1, desired_level)


# Side note: calculation_version refers to which method of information gain to use. 
#  If 0, use Entropy (or Info Gain); if 1, use ME; if 2, use Gini Index
def DetermineTree(all_examples, total_labels, calculation_version, desired_level, attributes_to_use):
    root = Node()
    root.Labels_in_Branch = total_labels
    root.Examples_in_Branch = all_examples
    root.Remaining_Attributes = attributes_to_use

    SplitNode(root, calculation_version, 0, desired_level)
    return root


## Method used to test tree on all examples provided and returns the error rate for this tree and data set
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
    correct_predictions = 0 # A prediction is 'correct' if the predicted label is the same as the examples true label (the label it came with)
    wrong_predictions = 0   # A prediction is 'wrong' if it is not the same as the true label
    for id in example_predictions:
        prediction = example_predictions[id]
        orig_example_label = examples_to_test[id].label
        if prediction == orig_example_label:
            correct_predictions += 1
        else:
            wrong_predictions += 1

    error = wrong_predictions / len(examples_to_test)
    return error


# Returns predicted label for the given example using the given tree for traversal and prediction
def TestExample(tree, example):
    current_node = tree
    prediction_from_tree = None
    while prediction_from_tree == None:
        attr_to_follow = current_node.Attribute_Split_With
        if attr_to_follow == "":
            prediction_from_tree = current_node.Prediction
        else:
            ex_value = example.attributes[attr_to_follow]
            current_node = current_node.node_list[ex_value]
    return prediction_from_tree


# Method that converts numerical attributes to binary attributes. Based on my glance through the data, the continuous numerical atts
#  are never 'unknown', that is there is never a ? for the value of these attributes.
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


# Creates from the parent set a subset of 1000 unique examples
#  No need to set label data because that is done when constructing the data subset for a subtree
def CreateSubsetWithoutReplacement(train_examples):
    new_sub_set = {}
    random_set = r.sample(range(1, len(train_examples)), 1000)
    new_id = 1
    for i in random_set:
        new_sub_set.update({new_id: train_examples[i]})
    return new_sub_set

#
def CreateDataSubsetWithReplacement(train_examples, desired_size, bank_labels):
    new_sub_set = {}
    newid = 1
    for x in range(desired_size):
        rand_id = r.randint(1, len(train_examples))
        new_sub_set.update({newid: train_examples[rand_id]})
        bank_labels.total_num_values += 1
        bank_labels.label_values_and_counts[train_examples[rand_id].label] += 1
        newid += 1
    return new_sub_set


########################################################################################################################################
# Also referred to as Recall
def TruePositiveRate(predictions):
    positives = 0
    negatives = 0
    rate = 0
    for pred in predictions:
        if pred == "1":
            positives += 1
        if pred == "0":
            negatives += 1
    rate = positives / (positives + negatives)
    return rate

def FalsePositiveRate(predictions):
    positives = 0
    negatives = 0
    rate = 0
    for pred in predictions:
        if pred == "1":
            positives += 1
        if pred == "0":
            negatives += 1
    rate = 1 - (negatives / (positives + negatives))
    return rate



# Global variables for the forests problem
DOING_FORESTS = False
RANDOM_LIMIT = 0


bank_attribute_values = {"age": {"over", "under"}, 
                         "workclass": {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", 
                                       "Without-pay", "Never-worked", "?"},
                         "fnlwgt": {"over", "under"},
                         "education": {"Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", 
                                       "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"}, 
                         "education-num": {"over", "under"},
                         "marital-status": {"Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", 
                                            "Married-AF-spouse", "?"}, 
                         "occupation": {"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", 
                                        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", 
                                        "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"},
                         "relationship": {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried", "?"},
                         "race": {"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"},
                         "sex": {"Female", "Male", "?"},
                         "capital-gain": {"over", "under"},
                         "capital-loss": {"over", "under"},
                         "hours-per-week": {"over", "under"},
                         "native-country": {"United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", 
                                            "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", 
                                            "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", 
                                            "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", 
                                            "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", 
                                            "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"}}

attributes_to_convert = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

def main():
    train_labels = Label_data(0, {"1": 0, "0": 0})
    test_labels = Label_data(0, {"1": 0, "0": 0})

    train_examples = {}
    test_examples = {}
    
    train_first = True
    test_first = True
    id = 1
    with open ( "KaggleData/train_final.csv" , 'r' ) as f: 
        for l in f:
            if train_first != True:
                terms = l.strip().split(',')
                attributes = {"age": terms[0], "workclass": terms[1], "fnlwgt": terms[2], "education": terms[3], "education-num": terms[4], 
                          "marital-status": terms[5], "occupation": terms[6], "relationship": terms[7], "race": terms[8], "sex": terms[9], 
                          "capital-gain": terms[10], "capital-loss": terms[11], "hours-per-week": terms[12], "native-country": terms[13]}
                temp = Example(id, attributes, terms[14]) 
                train_examples.update({id: temp})
                train_labels.total_num_values += 1
                train_labels.label_values_and_counts[temp.label] += 1
                id += 1
            else:
                train_first = False
            
    #id = 1
    with open( "KaggleData/test_final.csv", 'r') as f:
        for l in f:
            if test_first != True:
                terms = l.strip().split(',')
                attributes = {"age": terms[1], "workclass": terms[2], "fnlwgt": terms[3], "education": terms[4], "education-num": terms[5], 
                          "marital-status": terms[6], "occupation": terms[7], "relationship": terms[8], "race": terms[9], "sex": terms[10], 
                          "capital-gain": terms[11], "capital-loss": terms[12], "hours-per-week": terms[13], "native-country": terms[14]}
                temp = Example(int(terms[0]), attributes, None)
                test_examples.update({int(terms[0]): temp})
                # Not sure if I'll need these yet
                #test_labels.total_num_values += 1
                #test_labels.label_values_and_counts[temp.label] += 1
                #id += 1
            else:
                test_first = False

    TransformNumericals(train_examples, test_examples)

    ## Basic implementation plan, I will implement the AdaBoost and Bagging implementation from the homework here, and then figure out how
    ##  to compress the data I got into a prediction value for each example, then output that into a .csv 'Excel' file

    ### AdaBoost section


    ### Bagging section
    subtrees = {}
    #subtree_train_errors = {}
    #subtree_test_errors = {}
    id = 1
    for i in range(500):
        train_labels = Label_data(0, {"1": 0, "0": 0})
        subset = CreateDataSubsetWithReplacement(train_examples, 25000, train_labels)
        subtree = DetermineTree(subset, train_labels, 0, 14, ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                                                                  "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                                                                  "hours-per-week", "native-country"])
        #training_error = TestTree(subtree, train_examples)
        #test_error = TestTree(subtree, test_examples)
        #subtree_train_errors.update({id: training_error})
        #subtree_test_errors.update({id: test_error})
        subtrees.update({id: subtree})
        if id % 25 == 0:
            print("Finished the " + str(id) + "th tree")
        id += 1


    # In order to get the needed prediction, I think I need to predict on a example using each of the trees, then for that example
    #  compute the true positive rate or false positive rate depending on what the average prediction for that example is.
    prediction = {}
    for ex in test_examples:
        example = test_examples[ex]
        examples_predictions = []
        for t in subtrees:
            tree = subtrees[t]
            prediction = TestExample(tree, example)
            examples_predictions.append(prediction)
        # With all the predictions for this example, find the values for both rates, as I think they are both needed for AUROC.
        tpr = TruePositiveRate(examples_predictions)
        fpr = FalsePositiveRate(examples_predictions)

        # Do I just use the one that is higher as the report? I feel like I should always be using the TPR, but maybe I should just
        # Report the average? If the average is '1' should I have the prediction be 1.0? Or should I always report the TPR, regardless?
        prediction.update({example.example_number: tpr})

    ## With the predictions made, output to a .csv file for submission.
    with open("KaggleData/predictions.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["ID", "Prediction"]
        writer.writerow(header)
        for id in prediction:
            pred = prediction[id]
            line = [str(id), str(pred)]
            writer.writerow(line)

    print("File for submission has been completed.")


# Run program
main()