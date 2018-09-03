import pandas as pd
import numpy as np


class TreeNode:
    # TreeNode node to keep track of the different nodes
    def __init__(self,attribute, value: str, label: str = ""):
        self.attribute = attribute;
        self.value = value
        self.label = label
        self.nodes = {}

    @property
    def is_leaf(self):
        if len(self.nodes) == 0:
            return True
        return False

    def add_node(self, value, node):
        self.nodes[value] = node

    def get_node(self, feature):
        if feature in self.nodes:
            return
        assert isinstance(feature, str)
        return self.nodes[feature]

    def to_string(self, level = 0):
        ret = "\t"*level+repr(self.attribute)+": "+repr(self.value)+": "+repr(self.label)+"\n"
        for node in self.nodes:
            ret += self.nodes[node].to_string(level+1)
        return ret

    def get_result(self,case):
        if self.is_leaf :
            return self.label
        for node in self.nodes:
            if self.nodes[node].value == case[self.nodes[node].attribute]:
                return self.nodes[node].get_result(case)
        return self.label




class ID3Tree:
    # ID3Tree class implementation of ID3 decision tree

    def __init__(self, examples, attributes, target_column, positive_label, negative_label,name):
        # initializes the ID3 with features and targets
        self.examples = examples
        self.attributes = attributes
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.target_column = target_column
        self.root_node = TreeNode("root", self.positive_label)
        self.name = name

    def train(self):
        self.root_node = self.reduce(self.examples, "root","root", self.attributes)

    def print(self):
        print(self.root_node.to_string())

    def reduce(self, examples, target_attribute,attribute_value, attributes):
        has_positive = False
        has_negative = False
        target_attribute_positive = 0
        target_attribute_negative = 0

        # count the positives and negatives
        for i in range(len(examples)):
            example = examples[i]
            positive = example[self.target_column] == self.positive_label
            if positive:
                has_positive = True
            else:
                has_negative = True

            # count the most common occurrence
            if target_attribute in example:
                if positive:
                    target_attribute_positive = target_attribute_positive + 1
                else:
                    target_attribute_negative = target_attribute_negative + 1

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

        attribute_to_split_on = None
        max_gain = 0.0
        for attribute in attributes:
            gain = self.get_gain(examples, attribute, attributes[attribute])
            if gain > max_gain:
                max_gain = gain
                attribute_to_split_on = attribute

        values_to_split_on = attributes[attribute_to_split_on]


        node = TreeNode(target_attribute,attribute_value)
        for value in values_to_split_on:
            examples_with_value = get_examples_with_value(examples, attribute_to_split_on, value)
            temp_attributes = dict(attributes)
            del temp_attributes[attribute_to_split_on]
            temp_node = self.reduce(examples_with_value,attribute_to_split_on,value,temp_attributes)
            node.add_node(value,temp_node)

        return node


    def get_gain(self, examples, column, target_values):
        positive_count = 0.0
        negative_count = 0.0

        for example in examples:
            if example[self.target_column] == self.positive_label:
                positive_count = positive_count + 1.0
            else:
                negative_count = negative_count + 1.0

        temp_entropy = entropy(positive_count, negative_count)
        total_count = positive_count + negative_count
        for value in target_values:
            [s, pos_count, neg_count] = self.get_entropy(examples, column, value)
            temp_entropy += ((pos_count + neg_count) / total_count) * s

        return temp_entropy

    def get_entropy(self, examples, column, value):
        positive_count = 0.0
        negative_count = 0.0
        for example in examples:
            if example[column] == value:
                if example[self.target_column] == self.positive_label:
                    positive_count = positive_count + 1.0
                else:
                    negative_count = negative_count + 1.0

        return entropy(positive_count, negative_count), positive_count, negative_count

    def get_result(self,case):
        return self.root_node.get_result(case)


def get_examples_with_value(examples, attribute, value):
    examples_with_value = []
    for example in examples:
        if example[attribute] == value:
            examples_with_value.append(example)

    return examples_with_value


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


titanic_train = pd.read_csv('../input/train.csv')

print(titanic_train.head(100))

positive = True
negative = False
x1 = 1
x2 = 2
target = "target"

and_00 = {x1:negative, x2:negative, target:negative}
and_01 = {x1:negative, x2:positive, target:negative}
and_10 = {x1:positive, x2:negative, target:negative}
and_11 = {x1:positive, x2:positive, target:positive}

examples_and = [and_00,and_01,and_10,and_11]

or_00 = {x1:negative, x2:negative, target:negative}
or_01 = {x1:negative, x2:positive, target:positive}
or_10 = {x1:positive, x2:negative, target:positive}
or_11 = {x1:positive, x2:positive, target:positive}

examples_or = [or_00,or_01,or_10,or_11]

xor_00 = {x1:negative, x2:negative, target:negative}
xor_01 = {x1:negative, x2:positive, target:positive}
xor_10 = {x1:positive, x2:negative, target:positive}
xor_11 = {x1:positive, x2:positive, target:negative}

examples_xor = [xor_00,xor_01,xor_10,xor_11]

attributes = {}

# fill out the attributes from example
for example in examples_and:
    for attribute in example:
        if attribute != target:
            if not attribute in attributes :
                attributes[attribute] = []
                attributes[attribute].append(example[attribute])
            else :
                if not example[attribute] in attributes[attribute] :
                    attributes[attribute].append(example[attribute])

and_tree = ID3Tree(examples_and,attributes,target,positive,negative,"and")
and_tree.train()
and_tree.print();

or_tree = ID3Tree(examples_or,attributes,target,positive,negative,"or")
or_tree.train()
or_tree.print();

xor_tree = ID3Tree(examples_xor,attributes,target,positive,negative,"xor")
xor_tree.train()
xor_tree.print();

trees = [and_tree,or_tree,xor_tree]
for tree in trees:
    for example in tree.examples:
        result = tree.get_result(example)
        print("{}: {} {} {}: {} = {} == {}".format(x1, example[x1], tree.name, x2, example[x2], example[target], result))

    print("\n\n")
