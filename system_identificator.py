import networkx as nx

def system_identification(graph, init):
    # system identification I guess
    rebuilt_graph = nx.Graph()
    
    visited_nodes = set()
    explored_edges = set()
    stack = []

    current = init
    visited_nodes.add(current)
    rebuilt_graph
    num_actions = 0
    
    # DFS, where we ensure that each edge has been explored.
    while True:
        found_unexplored_edge = False

        for neighbor in graph.neighbors(current):
            edge = frozenset({current, neighbor}) # frozenset = unordered set
            num_actions += 1
            if edge not in explored_edges:
                found_unexplored_edge = True

                explored_edges.add(edge)
                attr = graph[current][neighbor]
                rebuilt_graph.add_edge(current, neighbor, **attr)

                stack.append(current)
                current = neighbor

                if current not in visited_nodes:
                    visited_nodes.add(current)
                    rebuilt_graph.add_node(current)
                break # we don't need to go try other edges
        
        if not found_unexplored_edge:
            # Nothing left to explore
            if not stack:
                break
            current = stack.pop()

    return rebuilt_graph, num_actions