import pandas as pd
import numpy as np
from sklearn import tree

def remap_data(data_row,values) :
    row_values = []
    for data in data_row:
        row_values.append(values.index(data))

    return row_values

def get_correct_ratio(test_results,values) :
    test_positive_count = 0
    test_negative_count = 0
    for i in range(len(test_results)):
        if test_results[i] == values[i] :
            test_positive_count += 1
        else :
            test_negative_count += 1

    test_total_count = test_positive_count + test_negative_count

    test_prediction_rate = 0.0

    if test_total_count > 0 :
        test_prediction_rate = test_positive_count/test_total_count

    return test_prediction_rate

titanic_data = pd.read_csv('../../input/train.csv')

row = len(titanic_data.index)

#split the data set in two
dfs = np.split(titanic_data, [np.int64(4*row/5)], axis=0)
titanic_train = dfs[0]
titanic_test = dfs[1]

target = "Survived"

features = ["Pclass", "SibSp", "Parch","Age",'Fare']

#change EmbarkedData
embarked_values = titanic_train["Embarked"].unique().tolist()
embarked_row = titanic_train["Embarked"].values

embarked_row_value = remap_data(embarked_row,embarked_values)

emb_em = pd.Series(embarked_row_value)


#change sex data to binary
sex_row = titanic_train["Sex"].values
sex_values = titanic_train["Sex"].unique().tolist()
sex_row_value = remap_data(sex_row,sex_values)

titanic_data_features = titanic_train[features]

sem = pd.Series(sex_row_value)

titanic_data_features["SexValues"] = sem.values


titanic_data_features["EmbarkedValues"] = emb_em.values


titanic_data_features.fillna(-100, inplace=True)


data = []
for index, row in titanic_data_features.iterrows():
    data.append(row)

survived = titanic_train[target].values


titanic_tree = tree.DecisionTreeClassifier()
titanic_tree.fit(data,survived)

train_results = titanic_tree.predict(data)

train_prediction_rate = get_correct_ratio(train_results,survived)

print('{} : {}'.format("train_prediction_rate",train_prediction_rate))


################################################

#change EmbarkedData
test_embarked_row = titanic_test["Embarked"].values

test_embarked_row_value = remap_data(test_embarked_row,embarked_values)

test_emb_em = pd.Series(test_embarked_row_value)


#change sex data to binary
test_sex_row = titanic_test["Sex"].values

test_sex_row_value = remap_data(test_sex_row,sex_values)

titanic_test_features = titanic_test[features]

test_sem = pd.Series(test_sex_row_value)

titanic_test_features["SexValues"] = test_sem.values


titanic_test_features["EmbarkedValues"] = test_emb_em.values


titanic_test_features.fillna(-100, inplace=True)


test_data = []
for index, row in titanic_test_features.iterrows():
    test_data.append(row)

test_survived = titanic_test[target].values


test_results = titanic_tree.predict(test_data)

test_prediction_rate = get_correct_ratio(test_results,test_survived)

print('{} : {}'.format("test_prediction_rate",test_prediction_rate))
