from valit_q_testbed_helper import init_problem, find_path
from q_learning_functions import *
from valit_functions import *
import csv

# get example list
problem = open('problem_circles.txt')
problines = problem.readlines()
problem.close()
num_of_ex = len(problines)/3

dims = 20 # number of samples per axis
radius = 1 # neightborhood radius (1 = four-neighbors)
# examples = [10,12]
# exnum = examples[0] # example number

N = 1000

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
         "Fully-greedy Q-learning with convergence (No discounting, no stochastic approximation) w/ term action & term goal"),

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
        prev_shortest_path = 0
        inconsistent = False
        worst_path = None
        best_path = None
        
        for i in range (N):
            has_path, path_literal, length, elapsed_time, shortest_path = find_path(graph, p1index,p2index, algorithm, args)

             # record min/max
            path_length = len(path_literal)
            if best_path is None or path_length < best_path:
                best_path = path_length
            if worst_path is None or path_length > worst_path:
                worst_path = path_length

            if i > 0:
                if shortest_path != prev_shortest_path:
                    inconsistent = True # to be expected in ["prob_valit", "q_learning_stochastic_path", "q_learning_dc_path"]
            
            prev_shortest_path = shortest_path
            avg_time += elapsed_time
                
        avg_time = avg_time/N
        example_results.append({
            "algorithm": info,
            "avg_time": avg_time,
            "inconsistent": inconsistent,
            "best_path": best_path,
            "worst_path": worst_path
        })
        print(f"Example {ex}: {info} | avg_time={avg_time:.4f}s | path={shortest_path} | inconsistent={inconsistent}")

    # -----------------------------
    # Save CSV file for this example
    # -----------------------------
    csv_filename = f"example_{ex}_results_{N}_samples.csv"

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Avg Time", "Inconsistent?", "Best Path", "Worst Path"])

        for r in example_results:
            writer.writerow([
                r["algorithm"],
                r["avg_time"],
                r["inconsistent"],
                r["best_path"],
                r["worst_path"]
            ])

