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
    attribute_value
    
class Label_data:
    label_values_and_counts

class Attribute_Data:
    attribute_name
    attribute_values
    value_counts
