import heapq
from itertools import count
import networkx as nx

def model_free_dijkstra(graph, init, goal_region):
    # system identification I guess
    rebuilt_graph = nx.Graph()
    
    visited_nodes = set()
    explored_edges = set()
    stack = []

    current = init
    visited_nodes.add(current)
    rebuilt_graph
    
    # DFS, where we ensure that each edge has been explored.
    while True:
        found_unexplored_edge = False

        for neighbor in graph.neighbors(current):
            edge = frozenset({current, neighbor}) # frozenset = unordered set
            
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
    return dijkstra_path(rebuilt_graph, init, goal_region)

# Slightly modified version of nx.dijkstra_path
def dijkstra_path(graph, init, goal_region):
    paths = {source: [source] for source in {init}}
    found_goal = None
    G_succ = graph._adj  # For speed-up (and works for both directed and undirected graphs)

    push = heapq.heappush
    pop = heapq.heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    seen[init] = 0
    push(fringe, (0, next(c), init))
    i = 0
    num_actions = 0
    while fringe:
        i += 1
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v in goal_region:
            found_goal = v
            break
        for u, e in G_succ[v].items():
            num_actions += 1
            cost = graph.get_edge_data(u, v)['weight']
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]

    path = paths[found_goal]
    has_loop = False

    return i, num_actions, path, has_loop, 0.0, 0