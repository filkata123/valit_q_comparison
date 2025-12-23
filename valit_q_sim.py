from valit_q_testbed_helper import init_problem, find_path
from q_learning_functions import *
from valit_functions import *
import csv
import numpy as np 

# get example list
problem = open('problem_circles.txt')
problines = problem.readlines()
problem.close()
num_of_ex = len(problines)/3

dims = 20 # number of samples per axis
radius = 1 # neightborhood radius (1 = four-neighbors)
# examples = [10,12]
# exnum = examples[0] # example number

N = 2

for ex in range(int(num_of_ex)):
    graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, ex, dims, radius)

    algorithms = [
        # Note: if discount factor is 0.99, it doesn't work, even if learning rate is 0.999
        (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.5, 0.999), 
         "Normal Q-learning (reward, discounting, stochastic approximation (aplha = 0.5) and termination goal)"),

        (q_learning_path_reward, (graph, p1index, goal_indices), 
         "Normal Q-learning (reward, discounting, stochastic approximation and termination goal)"),

        (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 0.9999), 
         "Normal Q-learning (reward, higher discounting, stochastic approximation and termination goal)"),

        (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 0.999, 1),
         "No-discounting Q-learning"),

        (q_learning_path_reward, (graph, p1index, goal_indices, 1000, 500, 1, 1),
         "No-discounting, no stochastic approximation Q-learning"),
        
        (q_learning_path, (graph, p1index, goal_indices),
         "cost-based Q-learning (discounting, stochastic approximation and termination goal)"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 0.999, 1),
         "cost-based no-discounting Q-learning"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1),
         "cost-based Q-learning (No discounting, no stochastic approximation)"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0.1, True),
         "cost-based Q-learning (No discounting, no stochastic approximation) w/ term action & term goal"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0.1, True, False),
         "cost-based Q-learning (No discounting, no stochastic approximation, no term goal) w/ term action"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0.1, False, False),
         "cost-based Q-learning (No discounting, no stochastic approximation, no termination at all)"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 3000, 1, 1, 0.1, True, True, "random"),
         "Fully-random exploration Q-learning(No discounting, no stochastic approximation) w/ term action & term goal"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0.1, True, True, "greedy"),
         "Fully-greedy exploration Q-learning(No discounting, no stochastic approximation) w/ term action & term goal"),

        (q_learning_path, (graph, p1index, goal_indices, 1, int(4e5), 1, 1, 0.1, True, False, "random"),
         "One-episode random-exploration Q-learning(No discounting, no stochastic approximation) w/ term action only"),

        (q_learning_path, (graph, p1index, goal_indices, 1000, 500, 1, 1, 0.1, True, True, "greedy", True),
         "Fully-greedy Q-learning with convergence (No discounting, no stochastic approximation) w/ term action & term goal (best case)"),

        (q_learning_dc_path, (graph, p1index, goal_indices),
         "Don't care Q-learning"),

        (q_learning_stochastic_path, (graph, p1index, goal_indices),
         "Stochastic Q-learning (converging)"),

        (valit_path, (graph, p1index, goal_indices),
         "Value Iteration"),

        (random_valit_path, (graph, p1index, goal_indices, False),
         "Random Action Value Iteration"),

        (prob_valit, (graph, p1index, goal_indices),
         "Stochastic Value Iteration"),
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
        
        for i in range (N):
            has_path, path, goal_in_path, _ , elapsed_time, path_length, num_iterations_or_episodes, num_actions_taken, has_loop  = find_path(graph, p1index,p2index, algorithm, args)

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
                
        avg_time = np.average(time_array)
        var_time = np.var(time_array)
        std_time = np.std(time_array)

        avg_iter = np.average(iterations_array)
        var_iter = np.var(iterations_array)
        std_iter = np.std(iterations_array)

        avg_action_count = np.average(num_actions_array)
        var_action_count = np.var(num_actions_array)
        std_action_count = np.std(num_actions_array)

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
            "shortest_path": shortest_path,
            "longest_path": longest_path
        })
        print(f"Example {ex}: {info} | Goal reached consistently? -> {goal_reached_consistently} | Encountered loops? -> {loops_encountered} | avg_time={avg_time:.4f}s | var_time={var_time:.4f}s | std_time={std_time:.4f}s | avg_act_count={avg_action_count} | shortest_path={shortest_path}")

    # -----------------------------
    # Save CSV file for this example
    # -----------------------------
    csv_filename = f"example_{ex}_results_{N}_samples.csv"

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Goal reached","Loops encountered", "Avg Time", "Var Time", "STD Time", "Avg Iterations" ,"Var Iterations", "STD iterations", "Avg action count" ,"Var action count", "STD action count", "Shortest Path", "Longest Path"])

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
                r["shortest_path"],
                r["longest_path"]
            ])

