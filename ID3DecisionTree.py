import pandas as pd
import ID3 as ID3
import numpy as np

def get_percent_accuracy_from_tree(tree,examples_df):
    positive_count = 0.0
    negative_count = 0.0
    for index, row in examples_df.iterrows():
        result = tree.get_result(row)

        if row[tree.target_column] == result:
            positive_count += 1.0
        else :
            negative_count += 1.0

    total = positive_count + negative_count
    percent_accuracy = 0.0
    if total > 0 :
        percent_accuracy = positive_count/total

    return percent_accuracy


titanic_data = pd.read_csv('../input/train.csv')

titanic_data["Age"].fillna(-100, inplace=True)
print(titanic_data["Age"].tolist())
row = len(titanic_data.index)
dfs = np.split(titanic_data, [np.int64(row/2)], axis=0)
titanic_train = dfs[0]
titanic_test = dfs[1]

columns = titanic_train.columns.tolist()
print(titanic_train.head(10))
print(columns)
target = "Survived"

features = ["Survived","Pclass", "Sex", "SibSp", "Parch", "Embarked"]


attributes = {}
survived = []
# fill out the attributes from example
for fature in features:
    if fature != target:
        values = titanic_train[fature].unique()
        attributes[fature] = values
        print('{}:{}'.format(fature, values))
    else :
        survived = titanic_train[fature].unique()
        print('{}:{}'.format(fature, survived))


titanic_train_tree = ID3.ID3Tree(titanic_train,attributes,target,survived[0],survived[1],"titanic_train")
titanic_train_tree.train()
titanic_train_tree.print()




columns = titanic_test.columns.tolist()
print(titanic_test.head(10))
print(columns)

training_percent_accuracy = get_percent_accuracy_from_tree(titanic_train_tree, titanic_train_tree.examples)
print('{}: {}'.format("training_percent_accuracy",training_percent_accuracy))

testing_percent_accuracy = get_percent_accuracy_from_tree(titanic_train_tree, titanic_test)
print('{}: {}'.format("testing_percent_accuracy",testing_percent_accuracy))


