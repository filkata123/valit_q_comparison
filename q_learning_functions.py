import random
from collections import defaultdict
from prob_model import probability_model
import time

def god_eye_convergence_check(graph, Q, alpha, gamma, t_goal, goal_region):
    newQ = Q.copy()
    for n in graph.nodes:
        if t_goal and n in goal_region:
            continue
        for m in graph.neighbors(n):
            cost = graph[n][m]['weight']
            next_state = m

            next_neighbors = list(graph.neighbors(next_state))
            min_q_next = min([newQ.get((next_state, a), 1.0E10) for a in next_neighbors]) if next_neighbors else 0

            newQ[(n, m)] = (1-alpha)*newQ[(n, m)] + alpha * (cost + gamma * min_q_next)
    return newQ

def god_eye_convergence_check_stochastic(graph, Q, alpha, gamma, t_goal, goal_region):
    newQ = Q.copy()
    for n in graph.nodes:
        if t_goal and n in goal_region:
            continue
        for m in graph.neighbors(n):
            prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(n))))
            cost = newQ[(n, m)] * prob_success
            cost = cost + newQ[(n, n)] * prob_stay
            for o in graph.neighbors(m):
                if o != n: #make sure that the current node is not taken into account
                    cost = cost + newQ[(n, o)] * prob_other

            cost = cost + graph[n][m]['weight'] 
            next_state = m

            next_neighbors = list(graph.neighbors(next_state))
            min_q_next = min([newQ.get((next_state, a), 1.0E10) for a in next_neighbors]) if next_neighbors else 0

            newQ[(n, m)] = (1-alpha)*newQ[(n, m)] + alpha * (cost + gamma * min_q_next)
    return newQ

def q_learning_stochastic_path_new(graph, init, goal_region, episodes=1000, max_steps=500, alpha=1, gamma=1):
    Q = defaultdict(float) # getting a key from this dict that doesn't exist reutrns 0.0
    epsilon = 0.1
    convergence_threshold = 1e-4
    num_actions = 0

    for episode in range(episodes):
        state = init
        max_delta = 0

        for _ in range(max_steps):
            actions = list(graph.neighbors(state))
            if not actions:
                break

            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = min(actions, key=lambda action: Q[(state, action)])

            neighbors = list(graph.neighbors(state))
            prob_success, prob_stay, prob_other = probability_model(len(neighbors))
            r = random.random()

            if r <= prob_success:
                next_state = action
            elif r <= prob_success + prob_stay:
                next_state = state
            else:
                others = [n for n in neighbors if n != action]
                next_state = random.choice(others)

            cost = graph[state][action]['weight']

            next_actions = list(graph.neighbors(next_state))
            min_q_next = min(Q[(next_state, a)] for a in next_actions) if next_actions else 0.0

            old_q = Q[(state, action)]
            Q[(state, action)] += alpha * (cost + gamma * min_q_next - Q[(state, action)])

            delta = abs(Q[(state, action)] - old_q)
            if delta > max_delta:
                max_delta = delta

            num_actions += 1
            state = next_state
            if state in goal_region:
                break
        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        if max_delta < convergence_threshold:
            #print(f"Q-learning converged at episode {episode}")
            break
    

    path = [init]
    current = init
    visited = set()
    has_loop = False

    while current not in goal_region:
        actions = list(graph.neighbors(current))
        if not actions:
            #print("No neighbors found.")
            break
        action = min(actions, key=lambda a: Q[(current, a)])

        neighbors = list(graph.neighbors(current))
        prob_success, prob_stay, prob_other = probability_model(len(neighbors))
        r = random.random()

        if r <= prob_success:
            next_state = action
        elif r <= prob_success + prob_stay:
            next_state = current
        else:
            others = [n for n in neighbors if n != action]
            next_state = random.choice(others)

        if next_state in visited:
            has_loop = True

        visited.add(current)
        path.append(next_state)
        current = next_state
 
    return episode, num_actions, path, has_loop, 0.0, 0


# Compute solution path from Q-table
def q_learning_stochastic_path(graph, init, goal_region, episodes=1000, max_steps=500, alpha=1, gamma=1):
    # Add an edge from the goal state to itself with 0 weight (termination action)
    for goal in goal_region:
        graph.add_edge(goal, goal, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = 0
    
    # Epsilon decay
    epsilon = 0.1

    # Convergence criterion
    convergence_threshold = 1e-4

    num_actions = 0

    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        max_delta = 0
        
        for _ in range(max_steps):
            neighbors = list(graph.neighbors(state))
            if not neighbors:
                print("No neighbors found.")
                break

            if random.random() < epsilon:
                action = random.choice(neighbors)
            else:
                action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))
            cost = graph[state][action]['weight']

            # Need this to account for staying in the same state -> can't happen naturally
            prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(state)))) # get probabilities
            choice = random.random()
            if choice <= prob_success:
                next_state = action # successful transition
            elif choice > prob_success and choice <= prob_success + prob_stay:
                next_state = state # stay
            else:
                current_range = prob_success + prob_stay
                for o in graph.neighbors(state):
                    if o != action: # make sure that the desired action is not taken into account
                        if choice > current_range and choice <= current_range + prob_other:
                            next_state = o
                            break
                        else: current_range += prob_other

            next_neighbors = list(graph.neighbors(next_state))
            min_q_next = min([Q.get((next_state, a), 1.0E10) for a in next_neighbors]) if next_neighbors else 0

            old_q = Q[(state, action)]
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha * (cost + gamma * min_q_next)

            # Track maximum absolute change in Q-values per episodes
            delta = abs(Q[(state, action)] - old_q)
            if delta > max_delta:
                max_delta = delta
          
            num_actions += 1
            state = next_state
            if state in goal_region:
                break
        
        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        if max_delta < convergence_threshold:
            #print(f"Q-learning converged at episode {episode}")
            break

    # Extract path from learned Q-values
    path = [init]
    current = init
    has_loop = False
    visited = set()
    i = 0
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            #print("No neighbors found.")
            break
        desired = min(neighbors, key=lambda a: Q.get((current, a), float('inf')))
        prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(current)))) # get probabilities
        choice = random.random()
        if choice <= prob_success:
            next_node = desired # successful transition
        elif choice > prob_success and choice <= prob_success + prob_stay:
            next_node = current # stay
        else:
            current_range = prob_success + prob_stay
            for o in graph.neighbors(current):
                if o != desired: # make sure that the desired node is not taken into account
                    if choice > current_range and choice <= current_range + prob_other:
                        next_node = o
                        break
                    else: current_range += prob_other
        if not has_loop and next_node in visited:
                has_loop = True # we don't break here since it is okay to have a loop in the probabilistic case
        path.append(next_node)
        current = next_node
        if i >= 100000: # we break here since we don't want to get stuck in an infinite loop
            break
        i += 1
    for goal in goal_region:
        graph.remove_edge(goal, goal) # clean up self-loop at goal
    return episode, num_actions, path, has_loop, 0.0, 0

# Compute solution path from Q-table
def q_learning_dc_path(graph, init, goal_region, episodes=15000, max_steps=5000, initial_epsilon=1):
    # Add an edge from the state to itself with 0 weight (stay cost)
    for n in graph.nodes:
        graph.add_edge(n, n, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = None # don't care value
    for goal in goal_region:
        Q[(goal, goal)] = 0.0 # goal state attractor. Works only in DC q-learning here since it's better than all other values.

    num_actions = 0
    # Epsilon decay
    epsilon = 0.1 # = initial_epsilon

    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        
        for _ in range(max_steps):
            neighbors = list(graph.neighbors(state))
            if not neighbors:
                print("No neighbors found.")
                break
            
            #TODO: Use digits of Pi in base n, where n = |actions| (sagemath)
            # if random.random() < epsilon:
            #     action = random.choice(neighbors)
            # else:
            valid_neighbors = [a for a in neighbors if Q.get((state, a)) is not None]
            action = min(valid_neighbors, key=lambda a: Q.get((state, a), 0.0)) if valid_neighbors else random.choice(neighbors)

            cost = graph[state][action]['weight']
            next_state = action

            next_neighbors = list(graph.neighbors(next_state))
            valid_q_values = [Q[(next_state, a)] for a in next_neighbors if Q[(next_state, a)] is not None]
            min_q_next = min(valid_q_values) if valid_q_values else None

            if min_q_next is not None:
                Q[(state, action)] = cost + min_q_next

            num_actions += 1
            state = next_state
            if state in goal_region:
                break
    
    #print(Q)
    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    has_loop = False
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        valid_neighbors = [a for a in neighbors if Q.get((current, a)) is not None]
        if not valid_neighbors:
            print("No valid neighbors found in Q-table. No path to goal available.")
            break
        next_node = min(valid_neighbors, key=lambda a: Q.get((current, a), float('inf')))
        if next_node in visited:
            has_loop = True
            break # avoid loops
        path.append(next_node)
        current = next_node

    for n in graph.nodes:
        graph.remove_edge(n, n) # clean up self-loops

    return episode, num_actions, path, has_loop, 0.0, 0

# Compute solution path from Q-table
def q_learning_path(graph, init, goal_region, 
                    episodes=1000, max_steps=500, alpha=0.999, gamma=0.999, initial_values=0, 
                    t_action = False, t_goal = True,
                    exploration_policy = "epsilon-greedy", convergence = False, deterministic = False, epsilon = 0.1, god_eye_convergence = False):
    # Add an edge from the goal state to itself with 0 weight (termination action)
    if t_action:
        for goal in goal_region:
            graph.add_edge(goal, goal, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = initial_values

    # Convergence criterion
    convergence_threshold = 0.0

    num_actions = 0

    # episode_returns = []
    # return_window = 10  # number of episodes to compare
    # return_tol = 0.0    # exact equality is valid in deterministic setting
    convergence_check_time = 0.0
    converged_action = 0

    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        max_delta = 0

        #episode_return = 0.0

        for _ in range(max_steps):
            neighbors = list(graph.neighbors(state))
            if not neighbors:
                print("No neighbors found.")
                break
            
            if exploration_policy == "random":
                action = chooser.choose(neighbors) if deterministic else random.choice(neighbors)
            elif exploration_policy == "greedy":
                action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))
            else: # epsilon-greedy
                if random.random() < epsilon:
                    action = chooser.choose(neighbors) if deterministic else random.choice(neighbors)
                else:
                    action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))

            cost = graph[state][action]['weight']
            next_state = action
            #episode_return += cost

            next_neighbors = list(graph.neighbors(next_state))
            min_q_next = min([Q.get((next_state, a), 1.0E10) for a in next_neighbors]) if next_neighbors else 0

            old_q = Q[(state, action)]
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha * (cost + gamma * min_q_next)

            # Track maximum absolute change in Q-values per episodes
            delta = abs(Q[(state, action)] - old_q)
            if delta > max_delta:
                max_delta = delta
            
            num_actions += 1

            # Check when the algorithm actually convergence and report for statistics (does not affect algorithm)
            if god_eye_convergence:
                t = time.time()
                if (num_actions != 0 and num_actions % 25500 == 0):
                    q_values = god_eye_convergence_check(graph, Q, alpha, gamma, t_goal, goal_region)
                    if Q == q_values:
                        converged_action = num_actions
                convergence_check_time += time.time() - t
            
            state = next_state
            # print(Q)
            # input("Press nter")
            if t_goal and state in goal_region:
                break
        
        # episode_returns.append(episode_return)
        #
        # if convergence and len(episode_returns) >= return_window:
        #     recent = episode_returns[-return_window:]
        #     if max(recent) - min(recent) <= return_tol:
        #         break

        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        if convergence:
            if max_delta == convergence_threshold:
                #print(f"Q-learning converged at episode {episode}")
                break
    
    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    has_loop = False
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        next_node = min(neighbors, key=lambda a: Q.get((current, a), float('inf')))
        if next_node in visited:
            has_loop = True
            break # avoid loops
        path.append(next_node)
        current = next_node
    if t_action:
        for goal in goal_region:
            graph.remove_edge(goal, goal) # clean up self-loop at goal
    return episode, num_actions, path, has_loop, convergence_check_time, converged_action

# Chooser that uses digits of Pi to make choices. Works for any base up to 10.
class PiChooser:
    def __init__(self, base4_filename):
        # gotten in a different way
        with open(base4_filename, "r") as f:
            self.pi_digits = f.read(10000000).strip()
        self.index = 0  # keep track of which digit we’re on
        
    def choose(self, neighbors):
        if not neighbors:
            return None

        while True:
            digit = int(self.pi_digits[self.index])
            self.index = (self.index + 1) % len(self.pi_digits)
            if digit < len(neighbors) or len(neighbors) == 0:
                break
        
        # Use digit to pick a neighbor
        choice = neighbors[digit]
        return choice

chooser = PiChooser("pi_base4_100m.txt")

# Compute solution path from Q-table
def q_learning_path_reward(graph, init, goal_region, episodes=1000, max_steps=500, alpha=0.999, gamma=0.999, initial_values=0, deterministic = False):
    #for goal in goal_region:
    #   graph.add_edge(goal, goal, weight=0.0)
    # Populate Q-table with zeros
    Q = {}
    for u in graph.nodes:
        for v in graph.neighbors(u):
            Q[(u, v)] = initial_values

    # Epsilon decay
    epsilon = 0.1

    # Convergence criterion
    convergence_threshold = 1e-4

    num_actions = 0
    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        max_delta = 0
        
        for _ in range(max_steps):
            neighbors = list(graph.neighbors(state))
            if not neighbors:
                print("No neighbors found.")
                break

            if random.random() < epsilon:
                if deterministic:
                    action = chooser.choose(neighbors)
                else:
                    action = random.choice(neighbors)
            else:
                action = max(neighbors, key=lambda a: Q.get((state, a), 0))

            reward = -graph[state][action]['weight']
            next_state = action

            next_neighbors = list(graph.neighbors(next_state))
            max_q_next = max([Q.get((next_state, a), 0.0) for a in next_neighbors]) if next_neighbors else 0

            old_q = Q[(state, action)]
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha * (reward + gamma * max_q_next)

            # Track maximum absolute change in Q-values per episodes
            delta = abs(Q[(state, action)] - old_q)
            if delta > max_delta:
                max_delta = delta

            num_actions += 1
            state = next_state
            # print(Q)
            # input("Press nter")
            if state in goal_region:
                break
        

        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        # if max_delta < convergence_threshold: 
        #     print(f"Q-learning converged at episode {episode}")
        #     break
        
        # for epsilon decay
        # epsilon = max(0.05, initial_epsilon * decay_rate**episode)

    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    has_loop = False
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        next_node = max(neighbors, key=lambda a: Q.get((current, a), float('-inf')))
        if next_node in visited:
            has_loop = True
            break  # avoid loops
        path.append(next_node)
        current = next_node
    # for goal in goal_region:
    #     graph.remove_edge(goal, goal) # clean up self-loop at goal
    return episode, num_actions, path, has_loop, 0.0, 0