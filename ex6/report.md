# Experiment 6 Report

<center>于斐, 202200130119. Nov. 25, 2024</center>


## Description

The following report shows the results of *Decision Tree* experiment. [Python](https://python.org/) and [matplotlib](https://matplotlib.org/) is used to complete this experiemnt.

## Skeleton

The implementation code is based on [this link](https://poloclub.gatech.edu/cse6242/2015fall/hw4/tree.py), which is provided in the tutorial.

## Implementation

Our decision tree support **Entropy** and **Gini Index** for classification.

### Entropy

Entropy is a measure of impurity or uncertainty in a dataset. It originates from information theory and is used to quantify the amount of disorder in a dataset. In the context of decision trees, entropy helps evaluate how well a split reduces the impurity of the data.

The entropy $ H(S) $ of a dataset $ S $ with $ n $ distinct classes can be defined by:

$$
H(S) = - \sum_{i=1}^{n} p_i \log_2(p_i)
$$

where $ p_i = \frac{\text{Number of instances in class } i}{\text{Total number of instances}} $ denotes The proportion of instances in class $ i $.

$ H(S) = 0 $ iff the dataset is completely pure (all instances belong to one class), and $ H(S) $ is maximum when all classes are equally probable (perfect disorder).

### Gini Index

The Gini Index is another measure of impurity in a dataset. It is widely used in decision tree algorithms such as CART (Classification and Regression Trees). Unlike entropy, the Gini Index is simpler to compute but often yields similar results.


The Gini Index $ G(S) $ of a dataset $ S $ with $ n $ distinct classes can be defined by:

$$
G(S) = 1 - \sum_{i=1}^{n} p_i^2
$$

where $ p_i = \frac{\text{Number of instances in class } i}{\text{Total number of instances}} $ denotes the proportion of instances in class $ i $.

$ G(S) = 0 $ iff the dataset is completely pure (all instances belong to one class), and $ G(S) $ approaches its maximum value (close to 0.5) when classes are equally distributed.

In our experiment, we compare the accuracy of the 2 methods by cross validation.

Implementation code in Python is as follows:

```python
def entropy(self, dataset):
    class_counts = {}
    for instance in dataset:
        label = instance[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    total_instances = len(dataset)
    entropy_value = 0
    for count in class_counts.values():
        probability = count / total_instances
        entropy_value -= probability * np.log2(probability) if probability > 0 else 0
    return entropy_value

def gini_index(self, dataset):
    class_counts = {}
    for instance in dataset:
        label = instance[-1]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    total_instances = len(dataset)
    gini_value = 1
    for count in class_counts.values():
        probability = count / total_instances
        gini_value -= probability ** 2
    return gini_value
```

## Evaluation

We use 10-fold cross validation for evaluation. 