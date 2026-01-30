from valit_q_testbed_helper import init_problem, find_path
from q_learning_functions import *
from valit_functions import *
from dijkstra_functions import *
from learning_rate_functions import *
import csv
import numpy as np 
from itertools import product
import networkx as nx
from datetime import datetime

# get example list
problem = open('problem_circles.txt')
problines = problem.readlines()
problem.close()
num_of_ex = len(problines)/3

dims = 20 # number of samples per axis
radius = 1 # neightborhood radius (1 = four-neighbors)
# examples = [10,12]
# exnum = examples[0] # example number

def run_simulations():
    N = 100
    print("Start: " + str(datetime.now()))
    for ex in range(0, int(num_of_ex)):
        graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, ex, dims, radius)

        algorithms = [
            # # Reward-based Q-learning
            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.001, 0.01),
            #  "Normal Q-learning (reward, alpha = 0.001, gamma = 0.01 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.01, 0.001),
            #  "Normal Q-learning (reward, alpha = 0.01, gamma = 0.001 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.01, 0.01),
            #  "Normal Q-learning (reward, alpha = 0.01, gamma = 0.01 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.1, 0.1),
            #  "Normal Q-learning (reward, alpha = 0.1, gamma = 0.1 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.3, 0.1),
            #  "Normal Q-learning (reward, alpha = 0.3, gamma = 0.1 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.3, 0.5),
            #  "Normal Q-learning (reward, alpha = 0.3, gamma = 0.5 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.3, 0.6),
            #  "Normal Q-learning (reward, alpha = 0.3, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.6, 0.6), 
            #  "Normal Q-learning (reward, alpha = 0.6, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.9, 0.6), 
            #  "Normal Q-learning (reward, alpha = 0.9, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.6), 
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.9), 
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.9 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.9, -1e4), 
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.9 and termination goal, initial values = -1e4)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999, -1e4), 
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.999 and termination goal, initial values = -1e4)"),

            # # With values of either alpha or gamma below 0.3 and 0.6 respectively we need to greatly increase the steps and episodes to get consistent paths to goal
            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.2, 0.999),
            #  "Normal Q-learning (reward, alpha = 0.2, gamma = 0.999 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.5),
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.5 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999), 
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.999 and termination goal, initial values = 0)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999, 1e4), 
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.999 and termination goal, initial values = +1e4)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999, 1e7), 
            #  "Normal Q-learning (reward, alpha = 0.999, gamma = 0.999 and termination goal, initial values = +1e7)"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 1),
            #  "No-discounting Q-learning"),

            # (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 1, 1),
            #  "No-discounting, no stochastic approximation Q-learning"),

            # # Cost-based Q-learning
            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.001, 0.01),
            #  "cost-based  Q-learning (alpha = 0.001, gamma = 0.01 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.01, 0.001),
            #  "cost-based  Q-learning (alpha = 0.01, gamma = 0.001 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.01, 0.01),
            #  "cost-based  Q-learning (alpha = 0.01, gamma = 0.01 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.1, 0.1),
            #  "cost-based  Q-learning (alpha = 0.1, gamma = 0.1 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.3, 0.1),
            #  "cost-based  Q-learning (alpha = 0.3, gamma = 0.1 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.3, 0.5),
            #  "cost-based  Q-learning (alpha = 0.3, gamma = 0.5 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.3, 0.6),
            #  "cost-based  Q-learning (alpha = 0.3, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.6, 0.6), 
            #  "cost-based  Q-learning (alpha = 0.6, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.9, 0.6), 
            #  "cost-based  Q-learning (alpha = 0.9, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.6), 
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.6 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.9), 
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.9 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.2, 0.999),
            #  "cost-based  Q-learning (alpha = 0.2, gamma = 0.999 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.5),
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.5 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.1),
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.1 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.01),
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.01 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.001),
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.001 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.0001),
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.0001 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.9, 1e4), 
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.9 and termination goal, initial values = +1e4)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999, 1e4), 
            #  "cost-based  Q-learning (alpha = 0.999, gamma = 0.999 and termination goal, initial values = +1e4)"),
            
            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999),
            #  "cost-based Q-learning (alpha = 0.999, gamma = 0.999 and termination goal, initial values = 0)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999, -1e4), 
            #  "cost-based Q-learning (alpha = 0.999, gamma = 0.999 and termination goal, initial values = -1e4)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.999, -1e7), 
            #  "cost-based Q-learning (alpha = 0.999, gamma = 0.999 and termination goal, initial values = -1e7)"),
            
            # # Cost-based Q-learning no disocunting or no stochastic approximation 
            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 1),
            #  "cost-based Q-learning (no-discounting, alpha = 0.999)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1),
            #  "cost-based Q-learning (No discounting, no stochastic approximation)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0, True),
            #  "cost-based Q-learning (No discounting, no stochastic approximation) w/ term action & term goal"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0, True, False),
            #  "cost-based Q-learning (No discounting, no stochastic approximation, no term goal) w/ term action"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0, False, False),
            #  "cost-based Q-learning (No discounting, no stochastic approximation, no termination at all)"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "random"),
            #  "Fully-random exploration Q-learning(No discounting, no stochastic approximation) w/ term action & term goal"),

            # (q_learning_path, (graph,p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "random", False, True),
            #  "Fully-random (deterministic with pi) exploration Q-learning (No discounting, no stochastic approximation) w/ term action & term goal"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0, True, True, "greedy"),
            #  "Fully-greedy exploration Q-learning(No discounting, no stochastic approximation) w/ term action & term goal"),

            # (q_learning_path, (graph, p1index, goal_indices, 1, int(4e5), 1, 1, 0, True, False, "random"),
            #  "One-episode random-exploration Q-learning(No discounting, no stochastic approximation) w/ term action only"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0, True, True, "greedy", True),
            #  "Fully-greedy Q-learning with convergence (No discounting, no stochastic approximation) w/ term action & term goal (best case)"),

            # (q_learning_dc_path, (graph, p1index, goal_indices),
            #  "Don't care Q-learning"),

            # # Deterministic convergence checks
            (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "greedy", False, False, 0, True),
             "cost-based Q-learning true convergence (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 0"),

            (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "epsilon-greedy", False, False, 0.25, True),
             "cost-based Q-learning true convergence (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 0.25"),

            (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "epsilon-greedy", False, False, 0.5, True),
             "cost-based Q-learning true convergence (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 0.5"),

            (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "epsilon-greedy", False, False, 0.75, True),
             "cost-based Q-learning true convergence (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 0.75"),

            (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "epsilon-greedy", False, False, 0.9, True),
             "cost-based Q-learning true convergence (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 0.9"),
            
            (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "random", False, False, 1, True),
             "cost-based Q-learning true convergence (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 1"),

            (q_learning_path, (graph,p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "random", False, True, 1, True),
             "Fully-random (deterministic with pi) exploration Q-learning (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 1"),

            # (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0, True, True, "random", False, False, 1, True),
            #  "cost-based Q-learning true convergence (No discounting, no stochastic approximation) w/ term action & term goal, epsilon = 1, 500 -> 3000 steps"),

            # (q_learning_path, (graph, p1index, goal_indices, 1, int(4e5), 1, 1, 0, True, False, "random", False, False, 1, True),
            #  "One-episode random-exploration Q-learning true convergence (No discounting, no stochastic approximation) w/ term action only"),

            # # Stochastic convergence checks, prob model = 0.9
            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 1),
            #  "Stochastic-problem Q-learning (converging, no discounting, no stochastic approximation)"),

            # # N.B. At gamma 0.5 we sometimes get a very very slow result, so we do not test that
            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 0.6),
            # "Stochastic-problem Q-learning (converging, no stochastic approximation, gamma = 0.6)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 0.9),
            # "Stochastic-problem Q-learning (converging, no stochastic approximation, gamma = 0.9)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.2, 1),
            # "Stochastic-problem Q-learning (converging, no discounting, alpha = 0.2)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.7, 0.7),
            # "Stochastic-problem Q-learning (converging, alpha = 0.7, gamma = 0.7)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.9, 0.9),
            # "Stochastic-problem Q-learning (converging, alpha = 0.9, gamma = 0.9)"),

            # # Stochastic, Prob model = 0.99
            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0.1, False, 0.99),
            #  "Stochastic-problem (0.99 success) Q-learning (converging, no discounting, no stochastic approximation)"),

            # # N.B. At gamma 0.5 we sometimes get a very very slow result, so we do not test that
            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 0.6, 0.1, False, 0.99),
            # "Stochastic-problem (0.99 success) Q-learning (converging, no stochastic approximation, gamma = 0.6)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 0.9, 0.1, False, 0.99),
            # "Stochastic-problem (0.99 success) Q-learning (converging, no stochastic approximation, gamma = 0.9)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.2, 1, 0.1, False, 0.99),
            # "Stochastic-problem (0.99 success) Q-learning (converging, no discounting, alpha = 0.2)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.7, 0.7, 0.1, False, 0.99),
            # "Stochastic-problem (0.99 success) Q-learning (converging, alpha = 0.7, gamma = 0.7)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.9, 0.9, 0.1, False, 0.99),
            # "Stochastic-problem (0.99 success) Q-learning (converging, alpha = 0.9, gamma = 0.9)"),

            # # Stochastic convergence checks (prob model 0.9)

            # # Value iteration
            # (valit_path, (graph, p1index, goal_indices),
            # "Value Iteration"),

            # (valit_path, (graph, p1index, goal_indices, 0.8),
            # "Discounted Value Iteration - gamma = 0.8"),

            # (valit_path, (graph, p1index, goal_indices, 0.6),
            # "Discounted Value Iteration - gamma = 0.6"),

            # (valit_path, (graph, p1index, goal_indices, 0.5),
            # "Discounted Value Iteration - gamma = 0.5"),

            # (valit_path, (graph, p1index, goal_indices, 0.3),
            # "Discounted Value Iteration - gamma = 0.3"),

            # (valit_path, (graph, p1index, goal_indices, 0.1),
            # "Discounted Value Iteration - gamma = 0.1"),

            # (valit_path, (graph, p1index, goal_indices, 0.01),
            # "Discounted Value Iteration - gamma = 0.01"),

            # (valit_path, (graph, p1index, goal_indices, 0.0001),
            # "Discounted Value Iteration - gamma = 0.0001"),

            # (valit_path, (graph, p1index, goal_indices, 0.00001),
            # "Discounted Value Iteration - gamma = 0.00001"),

            # (valit_path, (graph, p1index, goal_indices, 0.000001),
            # "Discounted Value Iteration - gamma = 0.000001"),

            # # Prob valit
            # (prob_valit, (graph, p1index, goal_indices),
            # "Stochastic Value Iteration"),

            # (prob_valit, (graph, p1index, goal_indices, 0.8),
            # "Discounted Stochastic Value Iteration - gamma = 0.8"),

            # (prob_valit, (graph, p1index, goal_indices, 0.6),
            # "Discounted Stochastic Value Iteration - gamma = 0.6"),

            # # N.B. At gamma 0.5 we sometimes get a very very slow result, so we do not test that here as well

            # (random_valit_path, (graph, p1index, goal_indices, False),
            # "Random Action Value Iteration"),

            # (random_valit_path, (graph, p1index, goal_indices, False, 0.8),
            # "Random Action Discounted Value Iteration - gamma = 0.8"),

            # (random_valit_path, (graph, p1index, goal_indices, False, 0.6),
            # "Random Action Discounted Value Iteration - gamma = 0.6"),

            # (random_valit_path, (graph, p1index, goal_indices, False, 0.5),
            # "Random Action Discounted Value Iteration - gamma = 0.5"),

            # (random_valit_path, (graph, p1index, goal_indices, False, 0.5),
            # "Random Action Discounted Value Iteration - gamma = 0.5"),

            # (q_valit_path, (graph, p1index, goal_indices),
            # "Q-factor Value Iteration"),

            # (q_prob_valit, (graph, p1index, goal_indices),
            # "Q-factor Stochastic Value Iteration"),

            (model_free_dijkstra, (graph, p1index, goal_indices),
            "Model-free Dijkstra"),

            (model_free_valit, (graph, p1index, goal_indices),
            "Model-free Value Iteration"),

            (model_free_valit, (graph, p1index, goal_indices, True),
            "Model-free Synchronous Value Iteration"),
        ]

        example_results = []

        for (algorithm, args, info) in algorithms:
            avg_time = 0
            longest_path = None
            shortest_path = None
            goal_reached_consistently = True
            loops_encountered = False
            time_array = []
            iterations_array = []
            num_actions_array = []
            converge_actions_array= []
            goal_discovered_time_array = []
            goal_discovered_actions_array = []
            optimal_init_ctg_time_array = []
            optimal_init_ctg_actions_array = []
            
            for i in range (N):
                has_path, path, goal_in_path, _ , elapsed_time, path_length, num_iterations_or_episodes, num_actions_taken, has_loop, converged_at_action, _,_, additional_data = find_path(graph, p1index,p2index, algorithm, args)

                # record min/max
                if shortest_path is None or path_length < shortest_path:
                    shortest_path = path_length
                if longest_path is None or path_length > longest_path:
                    longest_path = path_length

                if goal_in_path == False:
                    goal_reached_consistently = False
                
                if has_loop:
                    loops_encountered = True

                time_array.append(elapsed_time)
                iterations_array.append(num_iterations_or_episodes)
                num_actions_array.append(num_actions_taken)
                if converged_at_action != 0: # track those that converge
                    converge_actions_array.append(converged_at_action)

                if additional_data[0] is not None:
                    goal_discovered_time_array.append(additional_data[0])
                if additional_data[1] is not None:
                    goal_discovered_actions_array.append(additional_data[1])
                if additional_data[2] is not None and additional_data[2] != 0.0: # exclude non-converging ones    
                    optimal_init_ctg_time_array.append(additional_data[2])
                if additional_data[3] is not None and additional_data[3] != 0:
                    optimal_init_ctg_actions_array.append(additional_data[3])

            if len(time_array) != 0:    
                avg_time = np.average(time_array)
                var_time = np.var(time_array)
                std_time = np.std(time_array)

                avg_iter = np.average(iterations_array)
                var_iter = np.var(iterations_array)
                std_iter = np.std(iterations_array)

                avg_action_count = np.average(num_actions_array)
                var_action_count = np.var(num_actions_array)
                std_action_count = np.std(num_actions_array)

                if converge_actions_array:
                    convergence_rate = len(converge_actions_array) / N
                    avg_convergence_action = np.average(converge_actions_array)
                    var_convergence_action = np.var(converge_actions_array)
                    std_convergence_action = np.std(converge_actions_array)
                else:
                    convergence_rate = 0.0
                    avg_convergence_action = 0.0
                    var_convergence_action =  0.0
                    std_convergence_action =  0.0

                if goal_discovered_time_array:
                    avg_goal_discovered_time = np.average(goal_discovered_time_array)
                    var_goal_discovered_time = np.var(goal_discovered_time_array)
                    std_goal_discovered_time = np.std(goal_discovered_time_array)
                else:
                    avg_goal_discovered_time = None
                    var_goal_discovered_time = None
                    std_goal_discovered_time = None

                if goal_discovered_actions_array:
                    avg_goal_discovered_actions = np.average(goal_discovered_actions_array)
                    var_goal_discovered_actions = np.var(goal_discovered_actions_array)
                    std_goal_discovered_actions = np.std(goal_discovered_actions_array)
                else:
                    avg_goal_discovered_actions = None
                    var_goal_discovered_actions = None
                    std_goal_discovered_actions = None

                if optimal_init_ctg_time_array:
                    avg_goal_optimal_init_ctg_time = np.average(optimal_init_ctg_time_array)
                    var_goal_optimal_init_ctg_time = np.var(optimal_init_ctg_time_array)
                    std_goal_optimal_init_ctg_time = np.std(optimal_init_ctg_time_array)
                else:
                    avg_goal_optimal_init_ctg_time = None
                    var_goal_optimal_init_ctg_time = None
                    std_goal_optimal_init_ctg_time = None

                if optimal_init_ctg_actions_array:
                    avg_goal_optimal_init_ctg_actions = np.average(optimal_init_ctg_actions_array)
                    var_goal_optimal_init_ctg_actions = np.var(optimal_init_ctg_actions_array)
                    std_goal_optimal_init_ctg_actions = np.std(optimal_init_ctg_actions_array)
                else:
                    avg_goal_optimal_init_ctg_actions = None
                    var_goal_optimal_init_ctg_actions = None
                    std_goal_optimal_init_ctg_actions = None

                example_results.append({
                    "algorithm": info,
                    "goal_reached_consistently": goal_reached_consistently,
                    "loops_encountered" : loops_encountered,
                    "avg_time": avg_time,
                    "var_time" : var_time,
                    "std_time" : std_time,
                    "avg_iter": avg_iter,
                    "var_iter" : var_iter,
                    "std_iter" : std_iter,
                    "avg_action_count": avg_action_count,
                    "var_action_count" : var_action_count,
                    "std_action_count" : std_action_count,
                    "avg_convergence_action": avg_convergence_action,
                    "var_convergence_action" : var_convergence_action,
                    "std_convergence_action" : std_convergence_action,
                    "convergence_rate" : convergence_rate,
                    "shortest_path": shortest_path,
                    "longest_path": longest_path,
                    "avg_goal_discovered_time" : avg_goal_discovered_time,
                    "var_goal_discovered_time" : var_goal_discovered_time,
                    "std_goal_discovered_time" : std_goal_discovered_time,
                    "avg_goal_discovered_actions" : avg_goal_discovered_actions,
                    "var_goal_discovered_actions" : var_goal_discovered_actions,
                    "std_goal_discovered_actions" : std_goal_discovered_actions,
                    "avg_goal_optimal_init_ctg_time" : avg_goal_optimal_init_ctg_time, 
                    "var_goal_optimal_init_ctg_time" : var_goal_optimal_init_ctg_time,
                    "std_goal_optimal_init_ctg_time" : std_goal_optimal_init_ctg_time,
                    "avg_goal_optimal_init_ctg_actions" : avg_goal_optimal_init_ctg_actions,
                    "var_goal_optimal_init_ctg_actions" : var_goal_optimal_init_ctg_actions,
                    "std_goal_optimal_init_ctg_actions" : std_goal_optimal_init_ctg_actions
                })
                print(f"Example {ex}: {info} | Goal reached consistently? -> {goal_reached_consistently} | Encountered loops? -> {loops_encountered} | avg_time={avg_time:.4f}s | var_time={var_time:.4f}s | std_time={std_time:.4f}s | avg_act_count={avg_action_count} | shortest_path={shortest_path}")
            else:
                example_results.append({
                    "algorithm": "MemoryError:" + str(info),
                    "goal_reached_consistently": False,
                    "loops_encountered" : True,
                    "avg_time": 0,
                    "var_time" : 0,
                    "std_time" : 0,
                    "avg_iter": 0,
                    "var_iter" : 0,
                    "std_iter" : 0,
                    "avg_action_count": 0,
                    "var_action_count" : 0,
                    "std_action_count" : 0,
                    "avg_convergence_action": 0,
                    "var_convergence_action" : 0,
                    "std_convergence_action" : 0,
                    "convergence_rate" : 0,
                    "shortest_path": 0,
                    "longest_path": 0,
                    "avg_goal_discovered_time" : None,
                    "var_goal_discovered_time" : None,
                    "std_goal_discovered_time" : None,
                    "avg_goal_discovered_actions" : None,
                    "var_goal_discovered_actions" : None,
                    "std_goal_discovered_actions" : None,
                    "avg_goal_optimal_init_ctg_time" : None, 
                    "var_goal_optimal_init_ctg_time" : None,
                    "std_goal_optimal_init_ctg_time" : None,
                    "avg_goal_optimal_init_ctg_actions" : None,
                    "var_goal_optimal_init_ctg_actions" : None,
                    "std_goal_optimal_init_ctg_actions" : None
                })
                print(f"Example {ex}: {info} | Error.")

        # -----------------------------
        # Save CSV file for this example
        # -----------------------------
        csv_filename = f"example_{ex}_results_{N}_samples.csv"

        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm", "Goal reached","Loops encountered", "Avg Time", "Var Time", "STD Time", "Avg Iterations" ,"Var Iterations", "STD iterations", "Avg action count" ,"Var action count", "STD action count", "Avg convergence action" ,"Var convergence action", "STD convergence action", "Convergence rate", "Shortest Path", "Longest Path", "Avg Time Goal Discovered", "Var Time Goal Discovered", "STD Time Goal Discovered", "Avg Actions Goal Discovered", "Var Actions Goal Discovered", "STD Actions Goal Discovered", "Avg Time Optimal Initial Cost2Go", "Var Time Optimal Initial Cost2Go", "STD Time Optimal Initial Cost2Go", "Avg Actions Optimal Initial Cost2Go", "Var Actions Optimal Initial Cost2Go", "STD Actions Optimal Initial Cost2Go"])

            for r in example_results:
                writer.writerow([
                    r["algorithm"],
                    r["goal_reached_consistently"],
                    r["loops_encountered"],
                    r["avg_time"],
                    r["var_time"],
                    r["std_time"],
                    r["avg_iter"],
                    r["var_iter"],
                    r["std_iter"],
                    r["avg_action_count"],
                    r["var_action_count"],
                    r["std_action_count"],
                    r["avg_convergence_action"],
                    r["var_convergence_action"],
                    r["std_convergence_action"],
                    r["convergence_rate"],
                    r["shortest_path"],
                    r["longest_path"],
                    r["avg_goal_discovered_time"],
                    r["var_goal_discovered_time"],
                    r["std_goal_discovered_time"],
                    r["avg_goal_discovered_actions"],
                    r["var_goal_discovered_actions"],
                    r["std_goal_discovered_actions"],
                    r["avg_goal_optimal_init_ctg_time"], 
                    r["var_goal_optimal_init_ctg_time"],
                    r["std_goal_optimal_init_ctg_time"],
                    r["avg_goal_optimal_init_ctg_actions"],
                    r["var_goal_optimal_init_ctg_actions"],
                    r["std_goal_optimal_init_ctg_actions"]
                ])
    print("End: " + str(datetime.now()))

def run_learning_rate_x_prob_model_sim():
    N = 100
    ex = 8 # simple problem 
    graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, ex, dims, radius)

    learning_rates = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
    prob_model_success = [0.5, 0.7, 0.9, 0.99, 0.999, 0.9999]
    epsilon = 0.9

    params_product = list(product(learning_rates, prob_model_success))

    example_results = []

    csv_filename = f"stochastic_learning_rate_results_{N}_samples.csv"

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Goal reached", "Avg Time", "Var Time", "STD Time", "Avg Iterations" ,"Var Iterations", "STD iterations", "Avg action count" ,"Var action count", "STD action count", "Avg convergence action" ,"Var convergence action", "STD convergence action", "Shortest Path", "Longest Path"])
        for alpha, prob_success in params_product:
            avg_time = 0
            longest_path = None
            shortest_path = None
            goal_reached_consistently = True
            time_array = []
            iterations_array = []
            num_actions_array = []
            converge_actions_array= []

            algorithm = q_learning_stochastic_path
            args = (graph, p1index, goal_indices, 5000, 500, alpha, 1, epsilon, True, prob_success)
            info = f"Stochastic-problem ({prob_success} success) Q-learning (converging, alpha = {alpha}, gamma = 1, epsilon = {epsilon})"

            for i in range(N):
                has_path, path, goal_in_path, _ , elapsed_time, path_length, num_iterations_or_episodes, num_actions_taken, has_loop, converged_at_action, _,_, _ = find_path(graph, p1index,p2index, algorithm, args)

                # record min/max
                if shortest_path is None or path_length < shortest_path:
                    shortest_path = path_length
                if longest_path is None or path_length > longest_path:
                    longest_path = path_length

                if goal_in_path == False:
                    goal_reached_consistently = False

                time_array.append(elapsed_time)
                iterations_array.append(num_iterations_or_episodes)
                num_actions_array.append(num_actions_taken)
                converge_actions_array.append(converged_at_action)

            avg_time = np.average(time_array)
            var_time = np.var(time_array)
            std_time = np.std(time_array)

            avg_iter = np.average(iterations_array)
            var_iter = np.var(iterations_array)
            std_iter = np.std(iterations_array)

            avg_action_count = np.average(num_actions_array)
            var_action_count = np.var(num_actions_array)
            std_action_count = np.std(num_actions_array)

            avg_convergence_action = np.average(converge_actions_array)
            var_convergence_action = np.var(converge_actions_array)
            std_convergence_action = np.std(converge_actions_array)

            example_results.append({
                "algorithm": info,
                "goal_reached_consistently": goal_reached_consistently,
                "avg_time": avg_time,
                "var_time" : var_time,
                "std_time" : std_time,
                "avg_iter": avg_iter,
                "var_iter" : var_iter,
                "std_iter" : std_iter,
                "avg_action_count": avg_action_count,
                "var_action_count" : var_action_count,
                "std_action_count" : std_action_count,
                "avg_convergence_action": avg_convergence_action,
                "var_convergence_action" : var_convergence_action,
                "std_convergence_action" : std_convergence_action,
                "shortest_path": shortest_path,
                "longest_path": longest_path
            })

            # Write immediately
            writer.writerow([
                info,
                goal_reached_consistently,
                avg_time, var_time, std_time,
                avg_iter, var_iter, std_iter,
                avg_action_count, var_action_count, std_action_count,
                avg_convergence_action, var_convergence_action, std_convergence_action,
                shortest_path, longest_path
            ])

            # Force write to disk
            f.flush()


            print(f"Example {ex}: {info} | Goal reached consistently? -> {goal_reached_consistently} | | avg_time={avg_time:.4f}s | var_time={var_time:.4f}s | std_time={std_time:.4f}s | avg_act_count={avg_action_count} | shortest_path={shortest_path}")

def run_decaying_learning_rate_x_prob_model_sim():
    N = 10
    ex = 2 # mid problem 
    graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, ex, dims, radius)
    reachable = nx.node_connected_component(graph, p1index)
    num_nodes = len(reachable)

    prob_model_success = [0.9, 0.7, 0.5]
    decay_rate_inverse = [inverse_time_decay]
    decay_rate_functions = [polynomial_decay, polynomial_decay_normalized, visit_count_decay]
    alpha_zeros = [1, 0.1]
    omegas = [1/8, 1/4, 3/8, 1/2, 3/4, 7/8, 1, 2]
    decay_rates = [1, 0.1, 0.01, 0.001]
    epsilon = 1

    sims = []
    for decay_rate_func, alpha_zero, omega, decay_rate in product(decay_rate_inverse, alpha_zeros, omegas, decay_rates):
        sims.append((decay_rate_func, alpha_zero, omega, decay_rate))

    for decay_rate_func, omega in product(decay_rate_functions, omegas):
        sims.append((decay_rate_func, omega))

    example_results = []

    csv_filename = f"stochastic_learning_rate_decay_results_{N}_samples.csv"

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Goal reached", "Avg Time", "Var Time", "STD Time", "Avg Iterations" ,"Var Iterations", "STD iterations", "Avg action count" ,"Var action count", "STD action count", "Avg convergence action" ,"Var convergence action", "STD convergence action", "Shortest Path", "Longest Path"])
        for prob_success in prob_model_success:
             for sim in sims:
                decay_rate_func = sim[0]
                # TEMPORARY
                if decay_rate_func not in [inverse_time_decay, polynomial_decay]:
                    continue

                alpha_zero, omega, decay_rate = 0,0,0
                avg_time = 0
                longest_path = None
                shortest_path = None
                goal_reached_consistently = True
                time_array = []
                iterations_array = []
                num_actions_array = []
                converge_actions_array= []

                algorithm = q_learning_stochastic_path
                if decay_rate_func is inverse_time_decay:
                    _, alpha_zero, omega, decay_rate = sim
                    if decay_rate != 0.01 or omega < 3/4:
                        continue
                    args = (graph, p1index, goal_indices, 20000, 500, 1, 1, epsilon, True, prob_success, decay_rate_func, (alpha_zero, decay_rate, omega), num_nodes)
                    info = f"Stochastic-problem ({prob_success} success) Q-learning (converging, alpha_zero = {alpha_zero}, gamma = 1, epsilon = {epsilon}, omega = {omega}, decay_rate = {decay_rate}, decay_function = {decay_rate_func.__name__})"
                else:
                    _, omega = sim
                    if omega > 1/2:
                        continue
                    args = (graph, p1index, goal_indices, 20000, 500, 1, 1, epsilon, True, prob_success, decay_rate_func, (omega,), num_nodes)
                    info = f"Stochastic-problem ({prob_success} success) Q-learning (converging, alpha = 1, gamma = 1, epsilon = {epsilon}, omega = {omega}, decay_function = {decay_rate_func.__name__})"
                
                for i in range(N):
                    has_path, path, goal_in_path, _ , elapsed_time, path_length, num_iterations_or_episodes, num_actions_taken, has_loop, converged_at_action, _, _, _ = find_path(graph, p1index,p2index, algorithm, args)

                    # record min/max
                    if shortest_path is None or path_length < shortest_path:
                        shortest_path = path_length
                    if longest_path is None or path_length > longest_path:
                        longest_path = path_length

                    if goal_in_path == False:
                        goal_reached_consistently = False

                    time_array.append(elapsed_time)
                    iterations_array.append(num_iterations_or_episodes)
                    num_actions_array.append(num_actions_taken)
                    converge_actions_array.append(converged_at_action)

                avg_time = np.average(time_array)
                var_time = np.var(time_array)
                std_time = np.std(time_array)

                avg_iter = np.average(iterations_array)
                var_iter = np.var(iterations_array)
                std_iter = np.std(iterations_array)

                avg_action_count = np.average(num_actions_array)
                var_action_count = np.var(num_actions_array)
                std_action_count = np.std(num_actions_array)

                avg_convergence_action = np.average(converge_actions_array)
                var_convergence_action = np.var(converge_actions_array)
                std_convergence_action = np.std(converge_actions_array)

                example_results.append({
                    "algorithm": info,
                    "goal_reached_consistently": goal_reached_consistently,
                    "avg_time": avg_time,
                    "var_time" : var_time,
                    "std_time" : std_time,
                    "avg_iter": avg_iter,
                    "var_iter" : var_iter,
                    "std_iter" : std_iter,
                    "avg_action_count": avg_action_count,
                    "var_action_count" : var_action_count,
                    "std_action_count" : std_action_count,
                    "avg_convergence_action": avg_convergence_action,
                    "var_convergence_action" : var_convergence_action,
                    "std_convergence_action" : std_convergence_action,
                    "shortest_path": shortest_path,
                    "longest_path": longest_path
                })

                # Write immediately
                writer.writerow([
                    info,
                    goal_reached_consistently,
                    avg_time, var_time, std_time,
                    avg_iter, var_iter, std_iter,
                    avg_action_count, var_action_count, std_action_count,
                    avg_convergence_action, var_convergence_action, std_convergence_action,
                    shortest_path, longest_path
                ])

                # Force write to disk
                f.flush()


                print(f"Example {ex}: {info} | Goal reached consistently? -> {goal_reached_consistently} | | avg_time={avg_time:.4f}s | var_time={var_time:.4f}s | std_time={std_time:.4f}s | avg_act_count={avg_action_count} | shortest_path={shortest_path}")


def run_stochastic_convergence_simulations():
    N = 10
    ex_num = [1,10,11]
    print("Start: " + str(datetime.now()))
    for ex in ex_num:
        graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, ex, dims, radius)
        reachable = nx.node_connected_component(graph, p1index)
        num_nodes = len(reachable)

        prob_model_success = [0.999, 0.99]#, 0.9, 0.7, 0.5]
        epsilons = [0, 1/4, 1/2, 3/4, 0.9, 1]

        csv_filename = f"example_{ex}_stochastic_results_{N}_samples.csv"

        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Algorithm", "Goal reached", "Avg Time", "Var Time", "STD Time", "Avg Iterations" ,"Var Iterations", "STD iterations", "Avg action count" ,"Var action count", "STD action count", "Avg convergence action" ,"Var convergence action", "STD convergence action", "Convergence rate", "Shortest Path", "Longest Path", "Avg Time Goal Discovered", "Var Time Goal Discovered", "STD Time Goal Discovered", "Avg Actions Goal Discovered", "Var Actions Goal Discovered", "STD Actions Goal Discovered", "Avg Time Optimal Initial Cost2Go", "Var Time Optimal Initial Cost2Go", "STD Time Optimal Initial Cost2Go", "Avg Actions Optimal Initial Cost2Go", "Var Actions Optimal Initial Cost2Go", "STD Actions Optimal Initial Cost2Go", "Convergence rate Optimal Initial Cost2Go"])

            algorithms = []
            for prob in prob_model_success:
                for epsilon in epsilons:
                    # algorithms.append(
                    # (q_learning_stochastic_path, (graph, p1index, goal_indices, 3000, 3000, 0.05, 1, epsilon, True, prob),
                    # f"Stochastic-problem ({prob} success) Q-learning (converging, alpha = 0.05, gamma = 1, epsilon = {epsilon})"))

                    # algorithms.append(
                    # (q_learning_stochastic_path, (graph, p1index, goal_indices, 3000, 3000, 0.1, 1, epsilon, True, prob),
                    # f"Stochastic-problem ({prob} success) Q-learning (converging, alpha = 0.1, gamma = 1, epsilon = {epsilon})"))

                    algorithms.append(
                    (q_learning_stochastic_path, (graph, p1index, goal_indices, 3000, 3000, 0.5, 1, epsilon, True, prob),
                    f"Stochastic-problem ({prob} success) Q-learning (converging, alpha = 0.5, gamma = 1, epsilon = {epsilon})"))

                    algorithms.append(
                    (q_learning_stochastic_path, (graph, p1index, goal_indices, 3000, 3000, 0.7, 1, epsilon, True, prob),
                    f"Stochastic-problem ({prob} success) Q-learning (converging, alpha = 0.7, gamma = 1, epsilon = {epsilon})"))

                    algorithms.append(
                    (q_learning_stochastic_path, (graph, p1index, goal_indices, 3000, 3000, 0.9, 1, epsilon, True, prob),
                    f"Stochastic-problem ({prob} success) Q-learning (converging, alpha = 0.9, gamma = 1, epsilon = {epsilon})"))

                    algorithms.append(
                    (q_learning_stochastic_path, (graph, p1index, goal_indices, 3000, 3000, 0.99, 1, epsilon, True, prob),
                    f"Stochastic-problem ({prob} success) Q-learning (converging, alpha = 0.99, gamma = 1, epsilon = {epsilon})"))

                    # # Only check with 0.75?? lower omega means slower convergence -> might be better
                    # algorithms.append(
                    # (q_learning_stochastic_path, (graph, p1index, goal_indices, 3000, 3000, 1, 1, epsilon, True, prob, polynomial_decay, (0.2,)),
                    # f"Stochastic-problem ({prob} success) Q-learning (converging, alpha = 1\num_steps_taken^0.2, gamma = 1, epsilon = {epsilon})"))

                algorithms.append(
                    (prob_valit, (graph, p1index, goal_indices, 1, None, None, prob),
                    f"Stochastic Async Value Iteration ({prob} success)"),
                )

                algorithms.append(
                    (prob_valit_sync, (graph, p1index, goal_indices, 1, None, None, prob),
                    f"Stochastic Value Iteration ({prob} success)"),
                )

            
            for (algorithm, args, info) in algorithms:
                avg_time = 0
                longest_path = None
                shortest_path = None
                goal_reached_consistently = True
                time_array = []
                iterations_array = []
                num_actions_array = []
                converge_actions_array= []
                goal_discovered_time_array = []
                goal_discovered_actions_array = []
                optimal_init_ctg_time_array = []
                optimal_init_ctg_actions_array = []
                
                for i in range (N):
                    has_path, path, goal_in_path, _ , elapsed_time, path_length, num_iterations_or_episodes, num_actions_taken, has_loop, converged_at_action, _,_, additional_data = find_path(graph, p1index,p2index, algorithm, args)

                    # record min/max
                    if shortest_path is None or path_length < shortest_path:
                        shortest_path = path_length
                    if longest_path is None or path_length > longest_path:
                        longest_path = path_length

                    if goal_in_path == False:
                        goal_reached_consistently = False

                    time_array.append(elapsed_time)
                    iterations_array.append(num_iterations_or_episodes)
                    num_actions_array.append(num_actions_taken)
                    if converged_at_action != 0: # track those that converge
                        converge_actions_array.append(converged_at_action)

                    if additional_data[0] is not None and additional_data[0] != 0.0:
                        goal_discovered_time_array.append(additional_data[0])
                    if additional_data[1] is not None and additional_data[1] != 0:
                        goal_discovered_actions_array.append(additional_data[1])
                    if additional_data[2] is not None and additional_data[2] != 0.0: # exclude non-converging ones    
                        optimal_init_ctg_time_array.append(additional_data[2])
                    if additional_data[3] is not None and additional_data[3] != 0:
                        optimal_init_ctg_actions_array.append(additional_data[3])
    
                avg_time = np.average(time_array)
                var_time = np.var(time_array)
                std_time = np.std(time_array)

                avg_iter = np.average(iterations_array)
                var_iter = np.var(iterations_array)
                std_iter = np.std(iterations_array)

                avg_action_count = np.average(num_actions_array)
                var_action_count = np.var(num_actions_array)
                std_action_count = np.std(num_actions_array)

                if converge_actions_array:
                    convergence_rate = len(converge_actions_array) / N
                    avg_convergence_action = np.average(converge_actions_array)
                    var_convergence_action = np.var(converge_actions_array)
                    std_convergence_action = np.std(converge_actions_array)
                else:
                    convergence_rate = 0.0
                    avg_convergence_action = 0.0
                    var_convergence_action =  0.0
                    std_convergence_action =  0.0

                if goal_discovered_time_array:
                    avg_goal_discovered_time = np.average(goal_discovered_time_array)
                    var_goal_discovered_time = np.var(goal_discovered_time_array)
                    std_goal_discovered_time = np.std(goal_discovered_time_array)
                else:
                    avg_goal_discovered_time = None
                    var_goal_discovered_time = None
                    std_goal_discovered_time = None

                if goal_discovered_actions_array:
                    avg_goal_discovered_actions = np.average(goal_discovered_actions_array)
                    var_goal_discovered_actions = np.var(goal_discovered_actions_array)
                    std_goal_discovered_actions = np.std(goal_discovered_actions_array)
                else:
                    avg_goal_discovered_actions = None
                    var_goal_discovered_actions = None
                    std_goal_discovered_actions = None

                if optimal_init_ctg_time_array:
                    optimal_init_ctg_convergence_rate = len(optimal_init_ctg_time_array) / N
                    avg_goal_optimal_init_ctg_time = np.average(optimal_init_ctg_time_array)
                    var_goal_optimal_init_ctg_time = np.var(optimal_init_ctg_time_array)
                    std_goal_optimal_init_ctg_time = np.std(optimal_init_ctg_time_array)
                else:
                    optimal_init_ctg_convergence_rate = 0.0
                    avg_goal_optimal_init_ctg_time = None
                    var_goal_optimal_init_ctg_time = None
                    std_goal_optimal_init_ctg_time = None

                if optimal_init_ctg_actions_array:
                    avg_goal_optimal_init_ctg_actions = np.average(optimal_init_ctg_actions_array)
                    var_goal_optimal_init_ctg_actions = np.var(optimal_init_ctg_actions_array)
                    std_goal_optimal_init_ctg_actions = np.std(optimal_init_ctg_actions_array)
                else:
                    avg_goal_optimal_init_ctg_actions = None
                    var_goal_optimal_init_ctg_actions = None
                    std_goal_optimal_init_ctg_actions = None

                # Write immediately
                writer.writerow([
                    info,
                    goal_reached_consistently,
                    avg_time, var_time, std_time,
                    avg_iter, var_iter, std_iter,
                    avg_action_count, var_action_count, std_action_count,
                    avg_convergence_action, var_convergence_action, std_convergence_action,
                    convergence_rate,
                    shortest_path, longest_path,
                    avg_goal_discovered_time, var_goal_discovered_time, std_goal_discovered_time,
                    avg_goal_discovered_actions, var_goal_discovered_actions, std_goal_discovered_actions,
                    avg_goal_optimal_init_ctg_time, var_goal_optimal_init_ctg_time, std_goal_optimal_init_ctg_time,
                    avg_goal_optimal_init_ctg_actions, var_goal_optimal_init_ctg_actions, std_goal_optimal_init_ctg_actions,
                    optimal_init_ctg_convergence_rate
                ])

                # Force write to disk
                f.flush()


                print(f"Example {ex}: {info} | Goal reached consistently? -> {goal_reached_consistently} | | avg_time={avg_time:.4f}s | var_time={var_time:.4f}s | std_time={std_time:.4f}s | avg_act_count={avg_action_count} | shortest_path={shortest_path}")
    
    print("End: " + str(datetime.now()))

def main():
    #run_simulations()
    run_stochastic_convergence_simulations()
    #run_learning_rate_x_prob_model_sim()
    #run_decaying_learning_rate_x_prob_model_sim()

if __name__ == "__main__":
    main()
