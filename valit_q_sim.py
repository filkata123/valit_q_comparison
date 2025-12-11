from valit_q_testbed_helper import init_problem, find_path
from q_learning_functions import *
from valit_functions import *

# get example list
problem = open('problem_circles.txt')
problines = problem.readlines()
problem.close()
num_of_ex = len(problines)/3

dims = 20 # number of samples per axis
radius = 1 # neightborhood radius (1 = four-neighbors)
examples = [10,12]
exnum = examples[0] # example number

N = 10

graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, exnum, dims, radius)

algorithms = {
    valit_path : (graph, p1index, goal_indices),
    random_valit_path : (graph, p1index, goal_indices, False),
    prob_valit: (graph, p1index, goal_indices),
    q_learning_path_reward: (graph, p1index, goal_indices), 
    q_learning_path: (graph, p1index, goal_indices), #TODO: try variations as in obsidian
    q_learning_dc_path: (graph, p1index, goal_indices),
    q_learning_stochastic_path: (graph, p1index, goal_indices),
}

# TODO: Iterate over examples
for algorithm, args in algorithms.items():
    avg_time = 0
    for _ in range (N):
        has_path, path_literal, length, elapsed_time, shortest_path = find_path(graph, p1index,p2index, algorithm, args)

        avg_time += elapsed_time
    avg_time = avg_time/N
    print(f"{N} iterations of {algorithm.__name__} finds shortest path on average in {avg_time} seconds")




# print(has_path)
# print(path_literal)
# print(length)
# print(elapsed_time)
# print(shortest_path)