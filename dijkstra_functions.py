import heapq
from itertools import count
from system_identificator import system_identification
import networkx as nx

def model_free_dijkstra(graph, init, goal_region):
    rebuilt_graph, num_actions = system_identification(graph, init)
    results = list(dijkstra_path(rebuilt_graph, init, goal_region))
    results[1] += num_actions
    return results

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

    return i, num_actions, path, has_loop, 0.0, num_actions