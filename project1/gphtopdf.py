import networkx
import matplotlib.pyplot as plt

def gph_to_pdf(gph_file, pdf_file):
    '''Convert a .gph file to a .pdf file showing the directed graph.'''

    dag = networkx.DiGraph() # initialize empty graph
    # read the gph file and add edges to the graph
    with open(gph_file, 'r') as f:
        for line in f:
            parent, child = line.strip().split(',')
            # strip any whitespace from parent and child
            parent = parent.strip()
            child = child.strip()
            dag.add_edge(parent, child)

    pos = networkx.spring_layout(dag, k = 2.0, iterations = 50) # to make sure nodes are well spaced
    plt.figure(figsize=(10, 10))

    networkx.draw(dag, pos, with_labels=True, arrows=True, node_size=1000, node_color='lightpink', font_size=10, font_weight='bold')
    plt.savefig(pdf_file)
    print("Wrote graph visualization to {}".format(pdf_file))

def main():
    import sys
    if len(sys.argv) != 3:
        raise Exception("usage: python gphtopdf.py <infile>.gph <outfile>.pdf")

    gph_file = sys.argv[1]
    pdf_file = sys.argv[2]
    gph_to_pdf(gph_file, pdf_file)

if __name__ == "__main__":
    main()
