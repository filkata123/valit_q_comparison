import random
from prob_model import probability_model

# Compute solution path from Q-table
def q_learning_stochastic_path(graph, init, goal_region, episodes=1000, max_steps=500, alpha=1, gamma=1, initial_epsilon=1):
    # Add an edge from the goal state to itself with 0 weight (termination action)
    for goal in goal_region:
        graph.add_edge(goal, goal, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = -1.0E4 # TODO: investigate value, very sensitive to it
    
    # Epsilon decay
    epsilon = 0.1 # = initial_epsilon

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
                action = random.choice(neighbors) # in this case, the random action takes into account the stochasticity of the environment
            else:
                action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))
            cost = graph[state][action]['weight']
            next_state = action

            next_neighbors = list(graph.neighbors(next_state))
            min_q_next = min([Q.get((next_state, a), 1.0E10) for a in next_neighbors]) if next_neighbors else 0

            old_q = Q[(state, action)]
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha * (cost + gamma * min_q_next)

            # Track maximum absolute change in Q-values per episodes
            delta = abs(Q[(state, action)] - old_q)
            if delta > max_delta:
                max_delta = delta

            # Need this to account for staying in the same state -> can't happen naturally in 
            prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(state)))) # get probabilities
            choice = random.random()
            if choice <= prob_success:
                next_state = next_state # successful transition
            elif choice > prob_success and choice <= prob_success + prob_stay:
                next_state = state # stay
            else:
                current_range = prob_success + prob_stay
                for o in graph.neighbors(state):
                    if o != next_state: # make sure that the desired action is not taken into account
                        if choice > current_range and choice <= current_range + prob_other:
                            next_state = o
                            break
                        else: current_range += prob_other
                        
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
    while current not in goal_region:
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
        path.append(next_node)
        current = next_node
    for goal in goal_region:
        graph.remove_edge(goal, goal) # clean up self-loop at goal
    return episode, num_actions, path

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
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        valid_neighbors = [a for a in neighbors if Q.get((current, a)) is not None]
        if not valid_neighbors:
            print("No valid neighbors found in Q-table. No path to goal available.")
            break
        next_node = min(valid_neighbors, key=lambda a: Q.get((current, a), float('inf')))
        if next_node in visited:
            print("Loop detected in Q-table. No path to goal available.")
            break # avoid loops
        path.append(next_node)
        current = next_node

    for n in graph.nodes:
        graph.remove_edge(n, n) # clean up self-loops

    return episode, num_actions, path

# Compute solution path from Q-table
def q_learning_path(graph, init, goal_region, 
                    episodes=1000, max_steps=500, alpha=0.999, gamma=0.999, initial_epsilon=0.1, 
                    t_action = False, t_goal = True,
                    exploration_policy = "epsilon-greedy", convergence = False):
    # Add an edge from the goal state to itself with 0 weight (termination action)
    if t_action:
        for goal in goal_region:
            graph.add_edge(goal, goal, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = 1.0E4 # TODO: investigate value, very sensitive to it
    
    # Epsilon decay
    epsilon = initial_epsilon

    # Convergence criterion
    convergence_threshold = 0.0

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
            
            if exploration_policy == "random":
                action = random.choice(neighbors)
            elif exploration_policy == "greedy":
                action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))
            else: # epsilon-greedy
                if random.random() < epsilon:
                    action = random.choice(neighbors)
                else:
                    action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))

            cost = graph[state][action]['weight']
            next_state = action

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
            if t_goal and state in goal_region:
                break
        
        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        if convergence:
            if max_delta == convergence_threshold:
                #print(f"Q-learning converged at episode {episode}")
                break
    
    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        next_node = min(neighbors, key=lambda a: Q.get((current, a), float('inf')))
        if next_node in visited:
            #print("Loop detected in Q-table. No path to goal available.")
            break # avoid loops
        path.append(next_node)
        current = next_node
    if t_action:
        for goal in goal_region:
            graph.remove_edge(goal, goal) # clean up self-loop at goal
    return episode, num_actions, path

# Chooser that uses digits of Pi to make choices. Works for any base up to 10.
class PiChooser:
    def __init__(self, filename):
        # Load pi digits from file
        with open(filename, "r") as f:
            self.pi_digits = f.read().strip()
        self.index = 0  # keep track of which digit we’re on

    def choose(self, neighbors):
        if not neighbors:
            return None

        # Get next digit from pi (cycle back if at end)
        digit = int(self.pi_digits[self.index])
        self.index = (self.index + 1) % len(self.pi_digits)

        # Use digit to pick a neighbor
        choice = neighbors[digit % len(neighbors)]
        return choice

chooser = PiChooser("pi1k_base4.txt")

# Compute solution path from Q-table
def q_learning_path_reward(graph, init, goal_region, episodes=1000, max_steps=500, alpha=0.999, gamma=0.999, initial_epsilon=1, deterministic = False):
    #for goal in goal_region:
    #   graph.add_edge(goal, goal, weight=0.0)
    # Populate Q-table with zeros
    Q = {}
    for u in graph.nodes:
        for v in graph.neighbors(u):
            Q[(u, v)] = -1.0E4

    # Epsilon decay
    epsilon = 0.1 # = initial_epsilon
    # decay_rate = 0.9999

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
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        next_node = max(neighbors, key=lambda a: Q.get((current, a), float('-inf')))
        if next_node in visited:
            print("Loop detected in Q-table. No path to goal available.")
            break  # avoid loops
        path.append(next_node)
        current = next_node
    # for goal in goal_region:
    #     graph.remove_edge(goal, goal) # clean up self-loop at goal
    return episode, num_actions, path