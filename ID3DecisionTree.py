import pandas as pd
import ID3 as ID3
import numpy as np

def find_nearest(clusters,value):
    min = 0
    mind_dist = np.abs(clusters[min]-value)
    for i in range(len(clusters)):
        dist = np.abs(clusters[i] - value)
        if dist < mind_dist:
            min = i
            mind_dist = dist
    return min

def cluster_of_items(k,examples_df: pd.DataFrame ,attribute,iter = 100):
    array = examples_df[attribute].values
    uniques = examples_df[attribute].unique()
    klusters_center = []
    if len(uniques) < k:
        return uniques

    for i in range(k):
        klusters_center.append(uniques[i])
    for i in range(iter):
        klusters_mean = []
        klusters_count = []
        for i in range(k):
            klusters_mean.append(0)
            klusters_count.append(0)
        for value in array:
            cluster = find_nearest(klusters_center,value)
            klusters_mean[cluster] += value
            klusters_count[cluster] += 1
        for i in range(k):
            klusters_center[i] = klusters_mean[i]/klusters_count[i]

    return klusters_center

def cluster_data(clusters, examples_df: pd.DataFrame ,attribute):
    array = examples_df[attribute].values
    cluster_data = []

    for value in array:
        cluster = find_nearest(clusters,value)
        cluster_data.append(cluster)

    return cluster_data

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
dfs = np.split(titanic_data, [np.int64(3*row/4)], axis=0)
titanic_train = dfs[0]
titanic_test = dfs[1]

#age clusters
age_clusters = cluster_of_items(7,titanic_train,"Age")
cluster_age = cluster_data(age_clusters,titanic_train,"Age")
se = pd.Series(cluster_age)
titanic_train['ClusteredAge'] = se.values


cluster_age_test = cluster_data(age_clusters,titanic_test,"Age")
set = pd.Series(cluster_age_test)
titanic_test['ClusteredAge'] = set.values

#fare cluster
fare_clusters = cluster_of_items(3,titanic_train,"Fare")
cluster_fare = cluster_data(fare_clusters,titanic_train,"Fare")
se = pd.Series(cluster_fare)
titanic_train['ClusteredFare'] = se.values


cluster_fare_test = cluster_data(fare_clusters,titanic_test,"Fare")
set = pd.Series(cluster_fare_test)
titanic_test['ClusteredFare'] = set.values


print(fare_clusters)


columns = titanic_train.columns.tolist()
print(titanic_train.head(10))
print(titanic_train.head(10))
print(columns)
target = "Survived"

features = ["Survived","Pclass", "Sex", "SibSp", "Parch", "Embarked","ClusteredAge",'ClusteredFare']
features = ["Survived","Pclass", "Sex", "SibSp", "Parch", "Embarked","ClusteredAge",'ClusteredFare']


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


