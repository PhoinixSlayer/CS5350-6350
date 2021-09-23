#!/usr/bin/env python3

#import the-things-I-need

class Example:
    example_number
    attributes
    label

    def __init__(self, ex_number, attributes, lbl):
        self.example_number = ex_number
        self.attributes = attributes
        self.label = lbl

class Attribute:
    attribute_name
    num_each_value
    attribute_value_counts
    #attribute_values_neg

    def __init__(self, attr_name):
        self.attribute_name = attr_name          
        self.num_each_value = { }
        self.attribute_value_counts = { }   #make into a default dictionary, each key is attribute value, goes to array 0 <-> labels-1
     #   self.attribute_values_neg = { }   #make into a default dictionary (correct?)

    
class Label_data:
    total_num_labels
    label_values_and_counts

    def __init__(self, total_num, labels_and_values):
        self.total_num_values = total_num
        self.label_values_and_counts = labels_and_values

class Attribute_Data:
    attribute_name
    attribute_values
    value_counts


#Beginning of file that does the work I need


id = 1

print ("hello") #dummy output test string
