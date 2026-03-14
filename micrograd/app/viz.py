import os

# Add Graphviz to PATH
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Ensure graphs directory exists
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    print('making graph...')
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        # for any value in the graph, create a rectangular node
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

        if n._op:
            # if this value is a result of some operation, create an op node
            dot.node(name=uid + n._op, label=n._op)

            # connect this op node to the value node
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def save_graph(root, filename="graph", view=True):
    """Render and save graph to the graphs directory."""
    dot = draw_dot(root)
    filepath = os.path.join(GRAPHS_DIR, filename)
    dot.render(filepath, view=view)
    print(f"Graph saved to {filepath}.svg")
    return dot