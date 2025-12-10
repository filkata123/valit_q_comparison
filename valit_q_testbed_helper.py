
from ast import literal_eval
from rppl_util_necessary import *
import time

def init_problem(problines, exnum, dims, radius):
    obstacles = literal_eval(problines[exnum*3])
    initial = literal_eval(problines[exnum*3+1])
    goal = literal_eval(problines[exnum*3+2])

    # construct grid
    G = build_grid_graph(dims, radius, obstacles)
    # The next three lines delete the obstacle nodes (optional).
    #for i in range(len(G.nodes)):
    #        if point_inside_discs(G.nodes[i]['point'],obstacles):
    #            G.remove_node(i)

    p1index = find_closest_node(initial,G.nodes)
    p2index = find_closest_node(goal,G.nodes)
    # Print edge cost/weight
    # for (u,v,c) in G.edges().data():
    #     print("Edge (" + str(u) + ", " + str(v) +"): " + str(c))

    # Use a radius parameter to find the neighbors that will define the goal region
    goal_radius = 0
    goal_indices = list(nx.single_source_shortest_path_length(G, p2index, cutoff=goal_radius).keys())

    return G, p1index, p2index, obstacles, goal_indices


def find_path(graph, p1index, p2index, algorithm, args, kwargs = None):
    length = 0
    has_path = False
    #Since the graph is undirected, this is equivalent to checking if there is a path from p1index to any of the goal_indices
    if nx.has_path(graph,p1index,p2index):
        t = time.time()
        if kwargs == None:
            path = algorithm(*args)
        else:
            path = algorithm(*args, **kwargs)
        elapsed_time = time.time() - t
        for l in range(len(path)):
            if l > 0:
                if graph.get_edge_data(path[l],path[l-1]) is not None: # When there are loops, there is no weight in some cases
                    length += graph.get_edge_data(path[l],path[l-1])['weight']
        has_path = True

    return has_path, path, length, elapsed_time