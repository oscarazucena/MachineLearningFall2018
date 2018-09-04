import numpy as np
import pandas as pd
from enum import Enum
class TEST(Enum):
    EQUAL = 1
    LESS = 2
    GREAT = 3
class TestOperator:
    def __init__(self,operator : TEST):
        self.operator = operator
        self.value = None

    def set_value(self,value):
        self.value = value

    def test(self,value):
        if self.operator == TEST.EQUAL:
            return value == self.value
        if self.operator == TEST.LESS:
            return value <= self.value
        return value > self.value

class Test:
    def __init__(self, value, test_operator : TestOperator = TestOperator(TEST.EQUAL)):
        self.value = value
        self.test_operator = test_operator
        if self.test_operator.value == None:
            self.test_operator.set_value(value)

    def test(self,value):
        return self.test_operator.test(value)


class TreeNode:
    # TreeNode node to keep track of the different nodes
    def __init__(self,attribute, value: str, label: str = ""):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.nodes = {}

    @property
    def is_leaf(self):
    #property to tell if the node is a leaf or not
        if len(self.nodes) == 0:
            return True
        return False

    def add_node(self, value, node):
    #add_node adds a child node the current node
        self.nodes[value] = node

    def get_node(self, value):
    #get_node adds get a child node key by value
        if value in self.nodes:
            return
        return self.nodes[value]

    def to_string(self, level = 0):
    #to_string string representation with parent node printed first followed by child nodes
        ret = "\t"*level+repr(self.attribute)+": "+repr(self.value)+": "+repr(self.label)+"\n"
        for node in self.nodes:
            ret += self.nodes[node].to_string(level+1)
        return ret

    def get_result(self,case):
    #get_result returns the results from the input case
        if self.is_leaf :
            return self.label
        for node in self.nodes:
            if self.nodes[node].value == case[self.nodes[node].attribute]:
                return self.nodes[node].get_result(case)
        return self.label


class ID3Tree:
    # ID3Tree class implementation of ID3 decision tree

    def __init__(self, examples, attributes : pd.DataFrame, target_column, positive_label, negative_label,name):
        # initializes the ID3 with examples, attributes, target_column, positive_label, negative_label, and name
        self.examples = examples
        self.attributes = attributes
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.target_column = target_column
        self.root_node = TreeNode("root", self.positive_label)
        self.name = name


    def train(self):
    # train call this to train given the current exampples attribues
        self.root_node = self.reduce(self.examples, "root","root", self.attributes)

    def print(self):
        print(self.root_node.to_string())

    def reduce(self, examples, target_attribute,attribute_value, attributes):
        # reduce recursive function to call for training the tree

        # count the positives and negatives
        target_attribute_positive = examples[examples[self.target_column] == self.positive_label].count()[self.target_column]
        target_attribute_negative = examples[examples[self.target_column] == self.negative_label].count()[self.target_column]
        has_positive = target_attribute_positive > 0
        has_negative = target_attribute_negative > 0

        #print('{}: {}'.format("has_positive",has_positive))
        #print('{}: {}'.format("has_negative",has_negative))
        # if only positives or negatives are left or attributes are empty return with root
        if not (has_positive and has_negative) or len(attributes) == 0:
            if has_positive:
                return TreeNode(target_attribute,attribute_value, self.positive_label)

            if has_negative:
                return TreeNode(target_attribute,attribute_value, self.negative_label)

            max_key = self.positive_label
            if target_attribute_negative > target_attribute_positive:
                max_key = self.negative_label

            return TreeNode(target_attribute,attribute_value, max_key)

        #determine with attribute to split on
        attribute_to_split_on = self.find_attribute_to_split_on(attributes,examples)

        values_to_split_on = attributes[attribute_to_split_on]

        #recursively call reduce to train for each value in the target attribute
        node = TreeNode(target_attribute,attribute_value)
        for value in values_to_split_on:
            test = TestOperator(TEST.EQUAL)
            test.set_value(value)
            examples_with_value = get_examples_with_value(examples, attribute_to_split_on, value)
            temp_attributes = dict(attributes)
            #remove attribute to split on if is not numberic
            del temp_attributes[attribute_to_split_on]
            temp_node = self.reduce(examples_with_value,attribute_to_split_on,value,temp_attributes)
            node.add_node(value,temp_node)

        return node

    def find_attribute_to_split_on(self, attributes, examples):
        attribute_to_split_on = None
        max_gain = -1.0
        for attribute in attributes:
            gain = self.get_gain(examples, attribute, attributes[attribute])
            #print('{} {}:{}'.format(attribute, "gain", gain))
            if gain > max_gain:
                max_gain = gain
                attribute_to_split_on = attribute

        return attribute_to_split_on
    def get_gain_from_best_partition(self, attribute, examples):
        df = examples.sort_values(attribute)
        data_array = df[self.target_column].tolist()

        sequences = find_running_sequences(data_array)
        entropy, index = find_sequence_lowest_entropy(data_array,sequences,self.positive_label)
        value = (df.iloc[index][attribute]+df.iloc[index+1][attribute])/2.0
        return entropy, value

    def get_gain(self, examples, attribute, target_values):
    #returns the gain give the current examples, attribute to split on and attribute values
        positive_count = examples[examples[self.target_column] == self.positive_label].count()[self.target_column]
        negative_count = examples[examples[self.target_column] == self.negative_label].count()[self.target_column]

        temp_entropy = entropy(positive_count, negative_count)
        total_count = positive_count + negative_count
        for value in target_values:
            [s, pos_count, neg_count] = self.get_entropy(examples, attribute, value)
            temp_entropy -= ((pos_count + neg_count) / total_count) * s

        return temp_entropy

    def get_entropy(self, examples, attribute, value):
    #get_entropy returns entropy given the current examples, attribute and value
        attribute_df = examples[examples[attribute] == value]
        positive_count = attribute_df[attribute_df[self.target_column] == self.positive_label].count()[self.target_column]
        negative_count = attribute_df[attribute_df[self.target_column] == self.negative_label].count()[self.target_column]

        #print('{} : {}'.format('positive_count',positive_count))
        #print('{} : {}'.format('negative_count',negative_count))

        return entropy(positive_count, negative_count), positive_count, negative_count

    def get_result(self,case):
        #get_result returns results by iterating through each node give the case
        return self.root_node.get_result(case)


def get_examples_with_value(examples, attribute, value):
    examples_with_value = examples[examples[attribute] == value]
    return examples_with_value

def find_running_sequences(sorted_array):
    running_sequences = []
    for i in range(1,len(sorted_array)):
        if sorted_array[i-1] != sorted_array[i]:
            running_sequences.append(i-1)

    return running_sequences

def find_sequence_lowest_entropy(sorted_array,running_sequences, postive_value):
    running_count = []
    count = 0
    length = len(sorted_array)
    min_entropy = 1.0;
    min_entropy_index = 0
    for i in range(length):
        if postive_value == sorted_array[i]:
            count += 1
        running_count.append(count)
    for i in running_sequences:
        lower_positve_count = running_count[i]
        lower_negative_count = i-lower_positve_count
        top_positive_count = count-lower_positve_count
        top_negative_count = length- top_positive_count -lower_negative_count - lower_positve_count
        s1 = entropy(lower_positve_count,lower_negative_count)
        s2 = entropy(top_positive_count,top_negative_count)

        temp_entropy = ((lower_positve_count + lower_negative_count) / length) * s1 + ((top_positive_count + top_negative_count) / length) * s2
        if temp_entropy < min_entropy :
            min_entropy = temp_entropy
            min_entropy_index = i

    return temp_entropy,min_entropy_index



def entropy(positive_count, negative_count):
    positive_probability = 0.0
    negative_probability = 0.0
    if positive_count != 0:
        positive_probability = positive_count / (positive_count + negative_count)
    if negative_count != 0:
        negative_probability = negative_count / (positive_count + negative_count)

    temp_entropy = 0.0
    if positive_probability > 0:
        temp_entropy = -positive_probability * np.log2(positive_probability)
    if negative_probability > 0:
        temp_entropy += -negative_probability * np.log2(negative_probability)

    return temp_entropy
