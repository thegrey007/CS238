import numpy as np
import math
from collections import defaultdict
import networkx
from sklearn.metrics import mutual_info_score

def count_configurations(node, parents, data):
    """
    Count occurrences of each configuration of a node given its parents.

    Parameters:
    node (int): The index of the node.
    parents (list of int): List of parent node indices.
    data (np.ndarray): 2D numpy array.

    Returns:
    dict: item
    """
    # initialize a dictionary of dictionaries initialized to 0
    counts = defaultdict(lambda: defaultdict(int))
    # print("Counting configurations for node:", node, "with parents:", parents)
    for row in data:
        parent_values = tuple(row[parents]) if parents else ()
        node_value = row[node]
        # print("Row:", row, "Parent values:", parent_values, "Node value:", node_value)
        counts[parent_values][node_value] += 1
    return counts

def bayesian_score(graph, data):
    """
    Calculate the Bayesian score of a given graph structure based on the provided data.

    Parameters:
    graph (list of tuples): Each tuple is (node_index, [parent_indices])
    data (np.ndarray): 2D numpy array of shape (num_samples, num_variables)

    Returns:
    float: The Bayesian score of the graph.
    """

    score = 0.0
    for node, parents in graph:
        # print("Node:", node, "Parents:", parents)
        counts = count_configurations(node, parents, data)
        # count the number of unique values the node can take
        unique_node_values = set(data[:, node])
        # print("Unique values for node {}: {}".format(node, unique_node_values))
        for parent_values, val_dict in counts.items(): # all key value pairs - here it is key : (parent values), value : {node val: count}
            # print("Parent values:", parent_values, "Value counts:", val_dict)
            m_ij0 = sum(val_dict.values()) # total count of all values for this parent configuration
            for m_ijk in val_dict.values():
                score += math.lgamma(m_ijk + 1) - math.lgamma(1)  # Assuming uniform prior : alpha_ijk = 1 for all k
            score += math.lgamma(len(unique_node_values)) - math.lgamma(m_ij0 + len(unique_node_values))

    return score

def mutual_info_order(data):
    """
    Compute the mutual information between all pairs of variables and return an ordering based on it.
    We do this to get a better initial ordering for the K2 algorithm.

    Parameters:
    data (np.ndarray): 2D numpy array of shape (num_samples, num_variables)

    Returns:
    list: List of variable indices ordered by their mutual information.
    """
    num_vars = data.shape[1]
    mi_matrix = np.zeros((num_vars, num_vars))

    # Calculate mutual information for each pair of variables in given dataset
    for i in range(num_vars):
        for j in range(i + 1, num_vars): # mutual information is symmetric
            mi = mutual_info_score(data[:, i], data[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    # Sum mutual information for each variable
    mi_sums = np.sum(mi_matrix, axis=1)
    # Get ordering based on ascending mutual information sums
    order = np.argsort(mi_sums).tolist()
    
    return order