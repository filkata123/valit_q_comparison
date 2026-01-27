
from ast import literal_eval
from rppl_util_necessary import *
import time

from math import cos, sin, radians

def point_in_rotated_ellipse(p, center, a, b, angle_deg):
    """
    p      : (x, y) point
    center : (cx, cy) ellipse center
    a, b   : semi-major / semi-minor axes
    angle_deg : rotation angle in degrees
    """
    x, y = p
    cx, cy = center

    # translate
    xt = x - cx
    yt = y - cy

    # rotate
    theta = radians(angle_deg)
    xr =  xt * cos(theta) + yt * sin(theta)
    yr = -xt * sin(theta) + yt * cos(theta)

    # ellipse equation
    return (xr*xr)/(a*a) + (yr*yr)/(b*b) <= 1

def diagonal_ellipse_costs(G, cost=5, margin=0.15):
    """
    margin controls how close the ellipse gets to corners
    smaller margin => wider safe paths on both sides
    """

    # get bounds
    xs = [p[0] for _, p in G.nodes(data='point')]
    ys = [p[1] for _, p in G.nodes(data='point')]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    width  = xmax - xmin
    height = ymax - ymin

    # ellipse axes (shrink so it doesn't touch corners)
    a = (width) * (1 - margin)
    b = (height / 4)

    high_cost_nodes = {}

    for n, data in G.nodes(data=True):
        p = data['point']
        if point_in_rotated_ellipse(p, (cx, cy), a, b, -45):
            high_cost_nodes[n] = cost

    return high_cost_nodes

def init_problem(problines, exnum, dims, radius):
    obstacles = literal_eval(problines[exnum*3])
    initial = literal_eval(problines[exnum*3+1])
    goal = literal_eval(problines[exnum*3+2])

    # construct grid
    graph = build_grid_graph(dims, radius, obstacles)
    # The next three lines delete the obstacle nodes (optional).
    #for i in range(len(G.nodes)):
    #        if point_inside_discs(G.nodes[i]['point'],obstacles):
    #            G.remove_node(i)

    # center = (dims // 2) * dims + (dims // 2)
    # high_cost_nodes = {}
    # for dy in [-2, -1, 0, 1, 2]:
    #     for dx in [-2, -1, 0, 1, 2]:
    #         n = center + dy * dims + dx
    #         high_cost_nodes[n] = 50  # extra cost
    #vertex_costs = diagonal_ellipse_costs(graph, cost=50, margin=0.42)
    #apply_vertex_costs(graph,high_cost_nodes)

    p1index = find_closest_node(initial,graph.nodes)
    p2index = find_closest_node(goal,graph.nodes)
    # Print edge cost/weight
    # for (u,v,c) in G.edges().data():
    #     print("Edge (" + str(u) + ", " + str(v) +"): " + str(c))

    # Use a radius parameter to find the neighbors that will define the goal region
    goal_radius = 0
    goal_indices = list(nx.single_source_shortest_path_length(graph, p2index, cutoff=goal_radius).keys())

    return graph, p1index, p2index, obstacles, goal_indices


def find_path(graph, p1index, p2index, algorithm, args, kwargs = None):
    euclidean_distance = 0
    has_path = False
    goal_in_path = False
    path = {}
    path_length = 0.0
    elapsed_time = 0
    num_iterations_or_episodes = 0
    num_actions_taken = 0
    has_loop = False
    god_eye_convergence_time = 0.0
    converged_at_action = 0
    visits = {}  # Track state-action visits
    episode_trajectories = []
    #Since the graph is undirected, this is equivalent to checking if there is a path from p1index to any of the goal_indices
    if nx.has_path(graph,p1index,p2index):
        t = time.time()
        if kwargs == None:
            result = algorithm(*args)
        else:
            result = algorithm(*args, **kwargs)
        
        # Unpack result
        if len(result) == 7: # has visits
            num_iterations_or_episodes, num_actions_taken, path, has_loop, god_eye_convergence_time, converged_at_action, visits = result
        elif len(result) == 8 :
            num_iterations_or_episodes, num_actions_taken, path, has_loop, god_eye_convergence_time, converged_at_action, visits, episode_trajectories = result
        else:
            num_iterations_or_episodes, num_actions_taken, path, has_loop, god_eye_convergence_time, converged_at_action = result
        
        elapsed_time = time.time() - t
        elapsed_time = elapsed_time - god_eye_convergence_time
        path_length = len(path)
        if path_length != 0:
            has_path = True
            if p2index in path:
                goal_in_path = True

        for l in range(path_length):
            if l > 0:
                if graph.get_edge_data(path[l],path[l-1]) is not None: # When there are loops, there is no weight in some cases
                    euclidean_distance += graph.get_edge_data(path[l],path[l-1])['weight']
    
    return has_path, path, goal_in_path, euclidean_distance, elapsed_time, path_length, num_iterations_or_episodes, num_actions_taken, has_loop, converged_at_action, visits, episode_trajectories
