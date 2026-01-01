from valit_q_testbed_helper import init_problem, find_path
from q_learning_functions import *
from valit_functions import *
from dijkstra_functions import *
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

N = 100

def main():
    for ex in range(0, int(num_of_ex)):
        graph, p1index, p2index, obstacles, goal_indices = init_problem(problines, ex, dims, radius)

        algorithms = [
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

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 1),
            #  "Stochastic-problem Q-learning (converging, no discounting, no stochastic approximation)"),

            # # N.B. At gamma 0.5 we sometimes get a very very slow result, so we do not test that
            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 0.6),
            # "Stochastic-problem Q-learning (converging, no stochastic approximation, gamma = 0.6)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 1, 0.9),
            # "Stochastic-problem Q-learning (converging, no stochastic approximation, gamma = 0.6)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.2, 1),
            # "Stochastic-problem Q-learning (converging, no discounting, alpha = 0.2)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.7, 0.7),
            # "Stochastic-problem Q-learning (converging, alpha = 0.7, gamma = 0.7)"),

            # (q_learning_stochastic_path, (graph, p1index, goal_indices, 1000, 500, 0.9, 0.9),
            # "Stochastic-problem Q-learning (converging, alpha = 0.9, gamma = 0.9)"),

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
                    "shortest_path": 0,
                    "longest_path": 0
                })
                print(f"Example {ex}: {info} | Error.")

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

if __name__ == "__main__":
    main()
