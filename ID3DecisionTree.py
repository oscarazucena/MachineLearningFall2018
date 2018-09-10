import pandas as pd
import ID3 as ID3
import numpy as np
import matplotlib.pyplot as plt
import Booster as bs


def find_nearest(clusters,value):
    min = 0
    mind_dist = np.abs(clusters[min]-value)
    for i in range(len(clusters)):
        dist = np.abs(clusters[i] - value)
        if dist < mind_dist:
            min = i
            mind_dist = dist
    return min

def change_colum_type(examples_df: pd.DataFrame ,attribute,mapping):
    array = examples_df[attribute].values
    new_array = []
    uniques = examples_df[attribute].unique()

    if not len(uniques) == len(mapping):
        return

    for value in array:
        new_array.append(mapping[value])

    se = pd.Series(new_array)
    examples_df[attribute] = se.values


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

def cluster_labels(klusters_center):
    sorted = np.sort(klusters_center)
    print(sorted)
    labels = {}
    last = sorted[0]
    last_string = ""
    for i in range(1,len(sorted)):
        mid = (last+sorted[i])/2
        print(mid)
        k = find_nearest(klusters_center,last)
        labels[k] = last_string + " <= "+ repr(mid)
        last_string = repr(mid) + " > " + " and "
        last = sorted[i]

    k = find_nearest(klusters_center,last-.001)
    labels[k] = " > " + repr(mid)

    return labels

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
count = titanic_data["Age"].count()
div_count = int(count/8)
data_divs = [3*div_count,6*div_count,7*div_count]
dfs = np.split(titanic_data, data_divs, axis=0)
titanic_train = dfs[0]
titanic_stop_validate = dfs[1]
titanic_prune_validate = dfs[2]
titanic_test_all = dfs[3]

target = "Survived"

datasets = [titanic_train,titanic_prune_validate,titanic_stop_validate,titanic_test_all]
#create clusters
age_clusters = cluster_of_items(7,titanic_train,"Age")
fare_clusters = cluster_of_items(3,titanic_train,"Fare")
for dataset in datasets:
    cluster_age = cluster_data(age_clusters,dataset,"Age")
    se = pd.Series(cluster_age)
    dataset['ClusteredAge'] = se.values
    cluster_age_labels = cluster_labels(age_clusters)
    change_colum_type(dataset,'ClusteredAge',cluster_age_labels)

    cluster_fare = cluster_data(fare_clusters,dataset,"Fare")
    se = pd.Series(cluster_fare)
    dataset['ClusteredFare'] = se.values
    cluster_fare_labels = cluster_labels(fare_clusters)
    change_colum_type(dataset,'ClusteredFare',cluster_fare_labels)

    change_colum_type(dataset,target,{0:"Dead",1:"Survived"})


print(fare_clusters)


columns = titanic_train.columns.tolist()
print(titanic_train.head(10))
print(titanic_train.head(10))
print(columns)

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

max_tree = None
max_test_result = 0
tree_train_percent = []
tree_test_percent = []
tree_depths = []
for depth in range(8):
    tree_depths.append(depth+1)
    titanic_train_tree = ID3.ID3Tree(attributes,target,survived[0],survived[1],"titanic_train",depth)
    titanic_train_tree.fit(titanic_train)
    print('{}: {}'.format("max depth",titanic_train_tree.get_max_depth()))
    print('{}: {}'.format("number of nodes",titanic_train_tree.get_number_of_nodes()))

    columns = titanic_stop_validate.columns.tolist()

    training_percent_accuracy = get_percent_accuracy_from_tree(titanic_train_tree, titanic_train_tree.examples)
    print('{}: {}'.format("training_percent_accuracy",training_percent_accuracy))

    testing_percent_accuracy = get_percent_accuracy_from_tree(titanic_train_tree, titanic_stop_validate)
    print('{}: {}'.format("testing_percent_accuracy",testing_percent_accuracy))

    if max_test_result < testing_percent_accuracy:
        max_test_result = testing_percent_accuracy
        max_tree = titanic_train_tree

    tree_train_percent.append(training_percent_accuracy*100)
    tree_test_percent.append(testing_percent_accuracy*100)
    print('\n\n')


titanic_train_tree = max_tree


print('{}: {}'.format("Un-pruned number of nodes",titanic_train_tree.get_number_of_nodes()))
print('{}: {}'.format("Un-pruned max depth",titanic_train_tree.get_max_depth()))

unpruned_percent_accuracy = get_percent_accuracy_from_tree(titanic_train_tree, titanic_test_all)
print('{}: {}'.format("Un-prued percent accuracy",unpruned_percent_accuracy))

titanic_train_tree.prune(titanic_prune_validate)
print('{}: {}'.format("Pruned number of nodes",titanic_train_tree.get_number_of_nodes()))
print('{}: {}'.format("Pruned max depth",titanic_train_tree.get_max_depth()))

prune_percent_accuracy = get_percent_accuracy_from_tree(titanic_train_tree, titanic_test_all)
print('{}: {}'.format("pruting_percent_accuracy",prune_percent_accuracy))



for i in range(10):
    boosted_tree = bs.ID3BoostTree(attributes,target,survived[0],survived[1],"titanic_train",i)
    boosted_tree.fit(titanic_train)

    boosted_percent_accuracy = get_percent_accuracy_from_tree(boosted_tree, titanic_test_all)
    print('{}: {}'.format("boosted_percent_accuracy",boosted_percent_accuracy))
    print('{}: {}'.format("boosted_percent_tree_count",len(boosted_tree.trees)))
    print('{}'.format("alphas"))
    print(boosted_tree.alphas)

graph = titanic_train_tree.get_graph("titanic_train_tree.png")
graph.view()

graph_2 = titanic_train_tree.get_graph("titanic_train_tree_2.png")
graph_2.view()

plt.style.use('seaborn-whitegrid')
plt.plot(tree_depths, tree_train_percent, '-o', color='blue',label='Train % Accuracy')
plt.plot(tree_depths, tree_test_percent, '-+', color='red',label='Test % Accuracy')
plt.show()

