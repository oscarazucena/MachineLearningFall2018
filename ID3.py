import numpy as np
import pandas as pd
from enum import Enum
from graphviz import Digraph

class TreeNode:
    """
    Class for containing the decision nodes for a decision tree
    """
    def __init__(self,attribute, value, total_count, label = ""):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.count = total_count
        self.nodes = {}
        self.gain = 0

    @property
    def is_leaf(self):
        """
        property identify if tree is a leaf or not
        """
        if len(self.nodes) == 0:
            return True
        return False

    def set_gain(self,gain):
        """
        sets the node gain
        @param gain: node gain, calculate at time of training
        """
        self.gain = gain

    def set_positve_label_and_count(self,label,count):
        """
        sets the positive label and count
        @param label: label that has the positive outcome
        @param count: count of current positive outcome
        """
        self.positve_label = label
        self.positve_count = count

    def set_negative_label_and_count(self,label,count):
        """
        sets the negative label and count
        @param label: label that has the negative outcome
        @param count: count of current negative outcome
        """
        self.negative_label = label
        self.negative_count = count

    def add_node(self, value, node):
        """
        add new child node to this node
        :param value: key to address the node
        :param node:  child node to add
        """
        self.nodes[value] = node

    def get_node(self, value):
        """
        get child node base on value
        @param self:
        @param value: key to address node
        @return: the child node who's key is value
        """
        if value in self.nodes:
            return
        return self.nodes[value]

    def to_string(self, level = 0):
        """
        convert node to string further appending children if any
        @param level: lab level
        @return: string representing the current node
        """
        ret = "\t"*level+repr(self.attribute)+": "+repr(self.value)+": "+repr(self.label)+"\n"
        for node in self.nodes:
            ret += self.nodes[node].to_string(level+1)
        return ret

    def edge_name(self):
        """
        returns the edge name to use for visualing the node
        @return: the string represeting only the current node
        """
        ret = repr(self.attribute)+"=" + repr(self.value)
        ret += "\ncount="+repr(self.count)
        if not self.positve_label == None:
            ret += "\n" + repr(self.positve_label) + "=" + repr(self.positve_count)
        if not self.negative_label == None:
            ret += "\n" + repr(self.negative_label) + "=" + repr(self.negative_count)
        if not self.gain == None:
            ret += "\ngain="+repr(self.gain)
        if self.is_leaf:
            ret += "\nvalue="+repr(self.label)
        return ret

    def get_result(self,case):
        """
        returns the resulst of running the test case case
        :param case: test case to continue running
        :return: the results of running the test case including propatating down the the children
        """
        if self.is_leaf :
            return self.label
        for node in self.nodes:
            if self.nodes[node].value == case[self.nodes[node].attribute]:
                return self.nodes[node].get_result(case)
        return self.label

    def add_edge(self,g ,mine_name):
        count = 0
        g.node(mine_name, self.edge_name())
        for node in self.nodes:
            g.edge(mine_name,mine_name + repr(count))
            self.nodes[node].add_edge(g,mine_name + repr(count))
            count += 1

    def copy(self):
        copy = TreeNode( self.attribute, self.value, self.count,self.label)
        copy.set_gain(self.gain)
        copy.set_negative_label_and_count(self.negative_label,self.negative_count)
        copy.set_positve_label_and_count(self.positve_label,self.positve_count)
        for node in self.nodes:
            copy.add_node(node,self.nodes[node].copy())
        return copy

    def get_child_count(self):
        count = 0
        for node in self.nodes:
            count += 1 + self.nodes[node].get_child_count()
        return count
    def get_max_depth(self):
        """
        returns the max depth from the current to node to the next leaf node
        @return: the max depth from the current to node to the next leaf node
        """
        depth = 1
        for node in self.nodes:
            depth  = max(depth,1+self.nodes[node].get_max_depth())
        return depth

class ID3Tree:
    # ID3Tree class implementation of ID3 decision tree

    def __init__(self, attributes, target_column, positive_label, negative_label,name,stop_depth = 100):
        # initializes the ID3 with examples, attributes, target_column, positive_label, negative_label, and name
        self.attributes = attributes
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.target_column = target_column
        self.name = name
        self.debug_statement=False
        self.stop_depth = stop_depth

    def stop(self,current_depth):
        return current_depth >= self.stop_depth

    def fit(self,examples):
    # train call this to train given the current exampples attribues
        total_count = examples[self.target_column].count()
        self.root_node = TreeNode("root", self.positive_label,total_count)
        target_attribute_positive = examples[examples[self.target_column] == self.positive_label][self.target_column].count()
        target_attribute_negative = examples[examples[self.target_column] == self.negative_label][self.target_column].count()
        self.root_node.set_positve_label_and_count(self.positive_label,target_attribute_positive)
        self.root_node.set_negative_label_and_count(self.negative_label,target_attribute_negative)
        self.examples = examples
        self.root_node = self.reduce(examples, "root","root", self.attributes,0)

    def get_graph(self,filename):
        g = Digraph('G', filename=filename,format='pdf')
        self.root_node.add_edge(g,"0")
        return g

    def print_tree(self):
        """
        prints the information of the current three
        """
        print(self.root_node.to_string())

    def reduce(self, examples, target_attribute,attribute_value, attributes,current_depth,tab=1):
        # reduce recursive function to call for training the tree

        # count the positives and negatives
        target_attribute_positive = examples[examples[self.target_column] == self.positive_label][self.target_column].count()
        target_attribute_negative = examples[examples[self.target_column] == self.negative_label][self.target_column].count()
        total_count = examples[self.target_column].count()
        has_only_positive = target_attribute_positive == total_count
        has_only_negative = target_attribute_negative == total_count

        if self.debug_statement:
            print("\t"*tab+'{}: {}'.format("total_count",total_count))
            print("\t"*tab+'{}: {}'.format("target_attribute_positive",target_attribute_positive))
            print("\t"*tab+'{}: {}'.format("target_attribute_negative",target_attribute_negative))
            print("\t"*tab+'{}: {}'.format("has_only_positive",has_only_positive))
            print("\t"*tab+'{}: {}'.format("has_only_negative",has_only_negative))
        # if only positives or negatives are left or attributes are empty return with root

        if has_only_positive:
            node = TreeNode(target_attribute,attribute_value, total_count,self.positive_label)
            node.set_positve_label_and_count(self.positive_label,target_attribute_positive)
            node.set_negative_label_and_count(self.negative_label,target_attribute_negative)
            node.set_gain(0)
            return node

        if has_only_negative:
            node = TreeNode(target_attribute,attribute_value, total_count, self.negative_label)
            node.set_positve_label_and_count(self.positive_label,target_attribute_positive)
            node.set_negative_label_and_count(self.negative_label,target_attribute_negative)
            node.set_gain(0)
            return node

        max_key = self.positive_label
        if target_attribute_negative > target_attribute_positive:
            max_key = self.negative_label

        if len(attributes) == 0:
            node = TreeNode(target_attribute,attribute_value,total_count, max_key)
            node.set_positve_label_and_count(self.positive_label,target_attribute_positive)
            node.set_negative_label_and_count(self.negative_label,target_attribute_negative)
            node.set_gain(0)
            return node

        if self.stop(current_depth) :
            node = TreeNode(target_attribute,attribute_value,total_count, max_key)
            node.set_positve_label_and_count(self.positive_label,target_attribute_positive)
            node.set_negative_label_and_count(self.negative_label,target_attribute_negative)
            node.set_gain(0)
            return node

        if self.debug_statement:
            print("\t"*tab+'{}: {}'.format("continue: ",target_attribute))
        #determine with attribute to split on
        attribute_to_split_on,gain = self.find_attribute_to_split_on(attributes,examples)

        values_to_split_on = attributes[attribute_to_split_on]

        #recursively call reduce to train for each value in the target attribute
        node = TreeNode(target_attribute,attribute_value,total_count)
        node.set_positve_label_and_count(self.positive_label,target_attribute_positive)
        node.set_negative_label_and_count(self.negative_label,target_attribute_negative)
        node.set_gain(gain)

        for value in values_to_split_on:
            examples_with_value = get_examples_with_value(examples, attribute_to_split_on, value)
            temp_attributes = dict(attributes)
            #remove attribute to split on if is not numberic
            del temp_attributes[attribute_to_split_on]
            child_node = self.reduce(examples_with_value,attribute_to_split_on,value,temp_attributes,current_depth+1,tab+1)
            self.add_node(node,value,child_node)

        return node

    def add_node(self,parent,value,child):
        parent.add_node(value,child)


    def find_attribute_to_split_on(self, attributes, examples):
        attribute_to_split_on = None
        max_gain = -1.0
        for attribute in attributes:
            gain = self.get_gain(examples, attribute, attributes[attribute])
            #print('{} {}:{}'.format(attribute, "gain", gain))
            if gain > max_gain:
                max_gain = gain
                attribute_to_split_on = attribute

        return attribute_to_split_on,max_gain


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

    def get_number_of_nodes(self):
        return self.root_node.get_child_count()

    def get_max_depth(self):
        return self.root_node.get_max_depth()

    def copy(self):
        return self.root_node.copy()

    def prune(self, cross_validate):
        root_node_copy = self.copy()
        self.root_node = self.prune_helper(root_node_copy,cross_validate,root_node_copy)

    def prune_helper(self,node,cross_validate,root_node):
        all_childs_are_leaves = True
        if not node.is_leaf:
            for child_node in node.nodes:
                node.nodes[child_node] = self.prune_helper(node.nodes[child_node],cross_validate,root_node)
                all_childs_are_leaves = all_childs_are_leaves and node.nodes[child_node].is_leaf

        if all_childs_are_leaves:

            total = 0
            correct = 0
            for index, row in cross_validate.iterrows():
                result = self.get_result(row)
                if result == row[self.target_column]:
                    correct += 1
                total += 1
            percent_correct = 0
            if not total == 0:
                percent_correct = correct/total

            copy = node.copy()
            node.nodes = []
            node.label = self.positive_label
            total = 0
            correct = 0
            for index, row in cross_validate.iterrows():
                result = root_node.get_result(row)
                if result == row[self.target_column]:
                    correct += 1
                total += 1
            percent_correct_positive = 0
            if not total == 0:
                percent_correct_positive = correct/total

            node.label = self.negative_label
            total = 0
            correct = 0
            for index, row in cross_validate.iterrows():
                result = root_node.get_result(row)
                if result == row[self.target_column]:
                    correct += 1
                total += 1
            percent_correct_negative = 0
            if not total == 0:
                percent_correct_negative = correct/total

            if self.debug_statement:
                print('{}: {} {}'.format("pruning: ",node.attribute, node.value))
                print('{}: {}'.format("percent_correct_negative: ",percent_correct_negative))
                print('{}: {}'.format("percent_correct_positive: ",percent_correct_positive))
                print('{}: {}'.format("percent_correct: ",percent_correct))

            if percent_correct_negative >= percent_correct or percent_correct_positive >= percent_correct:
                if percent_correct_negative > percent_correct_positive:
                    node.value = self.negative_label
                else:
                    node.value = self.positive_label
            else:
                node = copy

        return node

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
