import ID3
import numpy as np

from ID3 import ID3Tree


class ID3BoostTree:
    def __init__(self,attributes, target_column, positive_label, negative_label,name,stop_depth = 100):
        # initializes the ID3 with examples, attributes, target_column, positive_label, negative_label, and name
        self.attributes = attributes
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.target_column = target_column
        self.name = name
        self.debug_statement = False
        self.stop_depth = stop_depth

    def fit(self,examples):
        """
        fit an Boosted ID3 tree according the examples passed in
        @param examples: examples to use creating the tree
        """
        self.trees = []
        self.alphas = []
        weights = []
        count = examples[self.target_column].count()
        weights = []
        for i in range(count):
            weights.append(1.0/count)
        self.reduce(examples,weights,self.attributes,0)

    def reduce(self, examples, weights, attributes,current_depth,tab=1):
        if current_depth == self.stop_depth:
            return

        count = examples[self.target_column].count()
        tree = self.create_tree(examples, attributes,"main")
        error = self.calculate_error(tree,examples,weights,self.positive_label)
        alpha = 0.5*np.log2((1.0-error)/error)
        weights = self.update_weights(examples,weights,tree,alpha,self.positive_label)
        self.reduce(examples,weights,attributes,current_depth+1,tab+1)
        self.trees.append(tree)
        self.alphas.append(alpha)

    def update_weights(self,examples,weights,tree,alpha,target):
        sum = 0.0
        for index, row in examples.iterrows():
            result = tree.get_result(row)
            result = (result == target)*1 - (result != target)*1
            value = row[self.target_column] == target
            weights[index] =  weights[index]*np.exp(-alpha*result*value)
            sum += weights[index]

        if sum > 0:
            for index, row in examples.iterrows():
                weights[index] /=  sum

        return weights

    def calculate_error(self,tree,examples, weights,target):
        error = 0.0
        for index, row in examples.iterrows():
            result = tree.get_result(row)
            result = (result == target)*1 - (result != target)*1
            value = row[self.target_column] == target
            error += weights[index]*result*value
        error = 0.5 - 0.5*error
        return error


    def create_tree(self, examples, attributes,name):
        tree: ID3Tree = ID3Tree(attributes,self.target_column,self.negative_label,self.positive_label,name,2)
        tree.fit(examples)
        return tree

    def get_result(self,case):
        result = 0.0
        for i in range(len(self.trees)):
            value = self.trees[i].get_result(case)
            if self.debug_statement:
                print('-------{} : {}'.format('value',value))
            value = (value == self.positive_label)*1 - (value != self.positive_label)*1
            if self.debug_statement:
                print('{} : {}'.format('value',value))
            result += self.alphas[i]*value
        label = self.positive_label
        if result < 0.0 :
            label = self.negative_label
        if self.debug_statement:
                print('{} : {}'.format('label',label))

        if self.debug_statement:
                print('{} : {}'.format('result',result))
        return label


def get_error(tree,row,target):
    result = tree.get_result(row)
    target = row[target]
    return result-target
