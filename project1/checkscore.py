import numpy as np
from utils import bayesian_score
import networkx
import sys

def check_score(datafile, graphfile):
    ''' utility function to check the score of a given graph in a gph file '''
    # load the data
    data = np.loadtxt(datafile, delimiter=',', skiprows=1, dtype=int)
    num_vars = data.shape[1]
    var_names = np.loadtxt(datafile, delimiter=',', dtype=str, max_rows=1)
    name2idx = {var_names[i]: i for i in range(num_vars)}
    dag = networkx.DiGraph() # initialize empty graph
    dag.add_nodes_from(range(num_vars))
    # read the gph file and add edges to the graph
    with open(graphfile, 'r') as f:
        for line in f:
            parent, child = line.strip().split(',')
            # strip any whitespace from parent and child
            parent = parent.strip()
            child = child.strip()
            dag.add_edge(name2idx[parent], name2idx[child])
    graph = [(i, list(dag.predecessors(i))) for i in range(num_vars)] # formatting for utils function
    score = bayesian_score(graph, data)
    print("Score of graph in {}: {}".format(graphfile, score))

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python checkscore.py <datafile>.csv <graphfile>.gph")

    datafile = sys.argv[1]
    graphfile = sys.argv[2]
    check_score(datafile, graphfile)

if __name__ == "__main__":
    main()