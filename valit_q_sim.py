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

graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, exnum, dims, radius)
print(graph)

has_path, path_literal, length, elapsed_time, shortest_path = find_path(graph, p1index,p2index, valit_path, (graph, p1index, goal_indices))

print(has_path)
print(path_literal)
print(length)
print(elapsed_time)
print(shortest_path)