import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, attribute=None, value=None, result=None):
        self.attribute = attribute  # Splitting attribute
        self.value = value          # Splitting value for the attribute
        self.children = {}          # Children nodes
        self.result = result        # Result if this node is a leaf node

def entropy(y):
    """Calculate entropy of a list of classes"""
    class_counts = Counter(y)
    entropy = 0
    total_instances = len(y)
    for count in class_counts.values():
        p = count / total_instances
        entropy -= p * np.log2(p)
    return entropy

def information_gain(X, y, attribute_index):
    """Calculate information gain of an attribute"""
    total_entropy = entropy(y)
    values, counts = np.unique(X[:, attribute_index], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset_y = y[X[:, attribute_index] == value]
        weighted_entropy += (count / len(y)) * entropy(subset_y)
    return total_entropy - weighted_entropy

def ID3(X, y, attributes):
    """Recursive function to build decision tree"""
    if len(set(y)) == 1:  # If all instances have the same class, return a leaf node
        return Node(result=y[0])
    
    if len(attributes) == 0:  # If no attributes left, return the majority class
        majority_class = Counter(y).most_common(1)[0][0]
        return Node(result=majority_class)
    
    max_information_gain = -1
    best_attribute = None
    for attribute_index in attributes:
        ig = information_gain(X, y, attribute_index)
        if ig > max_information_gain:
            max_information_gain = ig
            best_attribute = attribute_index
    
    node = Node(attribute=best_attribute)
    values = np.unique(X[:, best_attribute])
    new_attributes = [attr for attr in attributes if attr != best_attribute]
    for value in values:
        subset_X = X[X[:, best_attribute] == value]
        subset_y = y[X[:, best_attribute] == value]
        if len(subset_X) == 0:  # If subset is empty, return the majority class
            majority_class = Counter(y).most_common(1)[0][0]
            node.children[value] = Node(result=majority_class)
        else:
            node.children[value] = ID3(subset_X, subset_y, new_attributes)
    
    return node

def predict(root, instance):
    """Predict the class of an instance using the decision tree"""
    if root.result is not None:  # If it's a leaf node, return the result
        return root.result
    attribute_value = instance[root.attribute]
    if attribute_value not in root.children:  # If the attribute value is not in the tree, return None
        return None
    return predict(root.children[attribute_value], instance)

# Example usage:
# Sample dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# Convert data to dataframe
df = pd.DataFrame(data)

# Convert categorical values to numerical values
for column in df.columns:
    df[column] = pd.factorize(df[column])[0]

X = df.drop('Play', axis=1).values
y = df['Play'].values
attributes = list(range(X.shape[1]))

# Build the decision tree
root = ID3(X, y, attributes)

# Print the decision tree
def print_tree(node, depth=0):
    if node.result is not None:
        print(depth * '  ' + "Result:", node.result)
    else:
        print(depth * '  ' + "Attribute:", node.attribute)
        for value, child in node.children.items():
            print(depth * '  ' + "Value:", value)
            print_tree(child, depth + 1)

print_tree(root)

# Predict using the decision tree
instance = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': 'True'}
prediction = predict(root, instance)
print("Prediction:", prediction)
