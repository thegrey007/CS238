import sys
import math
import numpy as np
from utils import bayesian_score, mutual_info_order
import networkx
import time


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    # converting data csv to numpy array
    data = np.loadtxt(infile, delimiter=',', skiprows=1, dtype=int)

    # mapping variable names to indices to make computation faster (no more dicts)
    num_vars = data.shape[1]
    var_names = np.loadtxt(infile, delimiter=',', dtype=str, max_rows=1)
    name2idx = {var_names[i]: i for i in range(num_vars)}
    idx2names = {i: var_names[i] for i in range(num_vars)}

    # deciding an order for K2 algorithm
    order = list(range(num_vars))
    # initialize graph with no edges
    dag = networkx.DiGraph()
    dag.add_nodes_from(range(num_vars))
    # get the graph for the initial (no edges) structure
    graph = [(i, list(dag.predecessors(i))) for i in range(num_vars)]
    current_score = bayesian_score(graph, data)

    # time the algorithm
    start_time = time.time()
    
    # K2 algorithm
    for i in range(num_vars):
        node = order[i] # for each node in the order, try to add right children
        for potential_child in order[i+1:]: # only consider nodes that come after it in the order
            dag.add_edge(node, potential_child) # add the edge
            new_graph = [(j, list(dag.predecessors(j))) for j in range(num_vars)]
            new_score = bayesian_score(new_graph, data)
            if new_score > current_score: # if score improves, keep the edge and update score
                current_score = new_score
            else: # otherwise remove the edge
                dag.remove_edge(node, potential_child)

    # now, take this graph and do greedy hill climbing to improve it further
    for i in order:
        for j in range(num_vars):
            if i == j: # self loops not allowed
                continue
            # try adding edge i -> j if it doesn't create a cycle
            if not dag.has_edge(i, j):
                dag.add_edge(i, j)
                if networkx.is_directed_acyclic_graph(dag): # only keep if no cycle
                    new_graph = [(k, list(dag.predecessors(k))) for k in range(num_vars)]
                    new_score = bayesian_score(new_graph, data)
                    if new_score > current_score:
                        current_score = new_score
                    else:
                        dag.remove_edge(i, j)
                else:
                    dag.remove_edge(i, j)
            else:
                # try removing edge i -> j
                dag.remove_edge(i, j)
                new_graph = [(k, list(dag.predecessors(k))) for k in range(num_vars)]
                new_score = bayesian_score(new_graph, data)
                if new_score > current_score:
                    current_score = new_score
                else:
                    dag.add_edge(i, j)

    end_time = time.time()
    print("Time taken: {:.2f} seconds".format(end_time - start_time))
    print("Final score: {}".format(current_score))
    write_gph(dag, idx2names, outfile)
    print("Wrote graph to {}".format(outfile))


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    # time the compute function
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()