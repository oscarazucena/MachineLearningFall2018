import ID3 as ID3
import numpy as np
import pandas as pd

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

def get_x_and_y_or_z(x,y,z):
    return (x and y) or z

positive = True
negative = False
x1 = 1
x2 = 2
target = "target"

x1_arr = [negative,positive,negative,positive]
x2_arr = [negative,negative,positive,positive]

and_arr = [negative,negative,negative,positive]
and_df = pd.DataFrame({x1:x1_arr, x2:x2_arr, target:and_arr})


or_arr = [negative,positive,positive,positive]
or_df = pd.DataFrame({x1:x1_arr, x2:x2_arr, target:or_arr})

xor_arr = [negative,positive,positive,negative]
xor_df = pd.DataFrame({x1:x1_arr, x2:x2_arr, target:xor_arr})

columns = and_df.columns.tolist()

attributes = {}
# fill out the attributes from example
for column in columns:
    if column != target:
        values = and_df[column].unique()
        attributes[column] = values
        print('{}:{}'.format(column, values))


and_tree = ID3.ID3Tree(attributes,target,positive,negative,"and")
and_tree.fit(and_df)
and_tree.print()

or_tree = ID3.ID3Tree(attributes,target,positive,negative,"or")
or_tree.fit(or_df)
or_tree.print()

xor_tree = ID3.ID3Tree(attributes,target,positive,negative,"xor")
xor_tree.fit(xor_df)
xor_tree.print()

trees = [and_tree,or_tree,xor_tree]
for tree in trees:
    df = tree.examples
    for index, row in df.iterrows():
        result = tree.get_result(row)
        print("{}:{} {} {}:{} = {} == {}".format(x1, row[x1], tree.name, x2, row[x2], row[target], result))

    print("\n")


x_arr = [negative,positive,negative,positive,negative,positive,negative,positive]
y_arr = [negative,negative,positive,positive,negative,negative,positive,positive]
z_arr = [negative,negative,negative,negative,positive,positive,positive,positive]

x_and_y_or_z = []
for i in range(len(x_arr)):
    x_and_y_or_z.append(get_x_and_y_or_z(x_arr[i], y_arr[i], z_arr[i]))

x_and_y_or_z_df = pd.DataFrame({"x":x_arr, "y":y_arr, "z":z_arr, target:x_and_y_or_z})


columns = x_and_y_or_z_df.columns.tolist()
attributes = {}
# fill out the attributes from example
for column in columns:
    if column != target:
        values = x_and_y_or_z_df[column].unique()
        attributes[column] = values
        print('{}:{}'.format(column, values))

x_and_y_or_z_tree = ID3.ID3Tree(attributes,target,positive,negative,"x_and_y_or_z")
x_and_y_or_z_tree.fit(x_and_y_or_z_df)
x_and_y_or_z_tree.print()


x_and_y_or_z_tree_node_copy = x_and_y_or_z_tree.copy()
print(x_and_y_or_z_tree_node_copy.to_string())

sequence_test = [0, 0, 0, 1, 0, 0,0 ,1 , 1, 1, 1]

running_sequences = ID3.find_running_sequences(sequence_test)

print('{}: {}'.format("sequence_test",sequence_test))
print('{}: {}'.format("running_sequences",running_sequences))



graph = x_and_y_or_z_tree.get_graph("x_and_y_or_z_tree.png")

graph.view()
