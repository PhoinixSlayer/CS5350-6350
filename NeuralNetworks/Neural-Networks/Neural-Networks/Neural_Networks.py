### Code for solving the Neural Networks problems in Homework 5 ###


###
def main():
    train_examples = {}
    test_examples = {}
    
    id = 1
    label = 1
    with open ( "bank-note/train.csv" , 'r' ) as f: 
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3]) ]
            if float(terms[4]) == 0:
                label = -1
            else:
                label = 1
            temp = Example(id, attributes, label)  # Label of example, if 0 means forged, 1 means genuine
            train_examples.update({id: temp})
            id += 1
            
    id = 1
    label = 1
    with open( "bank-note/test.csv", 'r') as f:
        for l in f:
            terms = l.strip().split(',')
            attributes = [ 1, float(terms[0]), float(terms[1]), float(terms[2]), float(terms[3]) ]
            if float(terms[4]) == 0:
                label = -1
            else:
                label = 1 
            temp = Example(id, attributes, label)
            test_examples.update({id: temp})
            id += 1


##################################################
main()