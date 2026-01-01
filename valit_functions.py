from networkx.classes.function import get_node_attributes, set_node_attributes
import random
from prob_model import probability_model

# value iteration constants
failure_cost = 1.0E30
max_valits = 1000

def random_valit_path(graph, init, goal_region, epsilon_greedy = False, gamma = 1):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
        set_node_attributes(graph, {n:False}, 'updated') # has the node been visited
    for goal in goal_region:
        set_node_attributes(graph, {goal:0.0}, 'value') # This is the termination action, although it is not an action to speak of.
    
    # main loop
    num_actions = 0
    i = 0
    nodes_updated = get_node_attributes(graph, "updated")
    while i < max_valits:
        for m in graph.nodes:
            if not list(graph.neighbors(m)):
                continue 
            if not epsilon_greedy:
                chosen_n = random.choice(list(graph.neighbors(m)))
            else:
                if random.random() < 0.1:
                    chosen_n = random.choice(list(graph.neighbors(m)))
                else:
                    chosen_n = min(
                        graph.neighbors(m),
                        key=lambda n: graph.nodes[n]["value"] + graph.get_edge_data(n,m)['weight']
                    )
            step_cost = graph.get_edge_data(chosen_n,m)['weight']
            cost = gamma * graph.nodes[chosen_n]['value'] + step_cost
            num_actions += 1

            best_cost = failure_cost
            best_n = m
            if cost < best_cost:
                best_cost = cost
                best_n = chosen_n
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                current = graph.nodes[m].get('updated', None)
                set_node_attributes(graph, {m:not current}, 'updated')
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        if i != 0 and i % 50 == 0: # TODO: Can be optimized with statistics? If the node has 4 actions how many times do we need to visit it so that we are sure that all actions have been tried?
            new_nodes_updated =  get_node_attributes(graph, "updated")
            if nodes_updated == new_nodes_updated:
                break
            nodes_updated = new_nodes_updated
        i += 1
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        current_node = init
        has_loop = False
        visited = set()
        while not goal_reached:
            visited.add(current_node)
            nn = graph.nodes[current_node]['next']
            if nn in visited:
                has_loop = True
                break
            path.append(nn)
            current_node = nn
            if nn in goal_region:
                goal_reached = True
    #print("Stages: " + str(i))
    return i, num_actions, path, has_loop

def prob_valit(graph, init, goal_region, gamma = 1):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    for goal in goal_region:
        set_node_attributes(graph, {goal:0.0}, 'value')
    
    # main loop
    num_actions = 0
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            best_n = m
            
            for n in graph.neighbors(m):
                # Get edge count
                prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(m))))
                
                cost = gamma * graph.nodes[n]['value'] * prob_success # multiply by success probability
                cost = cost + gamma * graph.nodes[m]['value'] * prob_stay # multiply by stay probability

                # sum up expected costs and multiply by updated chosen probability
                for o in graph.neighbors(m):
                    if o != n: #make sure that the current node is not taken into account
                        num_actions += 1
                        cost = cost + gamma * graph.nodes[o]['value'] * prob_other

                # add weight to summed up cost    
                step_cost = graph.get_edge_data(n,m)['weight']
                cost = cost + step_cost

                if cost < best_cost:
                    best_cost = cost
                    best_n = n
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        i += 1
    
    #print(get_node_attributes(graph, 'value'))
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        current_node = init
        has_loop = False
        visited = set()
        j = 0
        while not goal_reached:
            visited.add(current_node)
            desired = graph.nodes[current_node]['next'] # select our desired node
            prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(current_node)))) # get probabilities
            choice = random.random()
            if choice <= prob_success:
                nn = desired # successful transition
            elif choice > prob_success and choice <= prob_success + prob_stay:
                nn = current_node # stay
            else:
                current_range = prob_success + prob_stay
                for o in graph.neighbors(current_node):
                    if o != desired: # make sure that the desired node is not taken into account
                        if choice > current_range and choice <= current_range + prob_other:
                            nn = o
                            break
                        else: current_range += prob_other
            if not has_loop and nn in visited:
                has_loop = True # we don't break here since it is okay to have a loop in the probabilistic case
            path.append(nn)
            current_node = nn
            if j >= 100000: # we break here since we don't want to get stuck in an infinite loop
                break
            j += 1
            if nn in goal_region:
                goal_reached = True
    #print("Stages: " + str(i))
    return i, num_actions, path, has_loop

def q_prob_valit(graph, init, goal_region, gamma = 1):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    for goal in goal_region:
        set_node_attributes(graph, {goal:0.0}, 'value')
    
    # main loop
    num_actions = 0
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            best_n = m
            
            Q = {}
            for n in graph.neighbors(m):
                # Get edge count
                prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(m))))
                
                cost = gamma * graph.nodes[n]['value'] * prob_success # multiply by success probability
                cost = cost + gamma * graph.nodes[m]['value'] * prob_stay # multiply by stay probability

                # sum up expected costs and multiply by updated chosen probability
                for o in graph.neighbors(m):
                    if o != n: #make sure that the current node is not taken into account
                        num_actions += 1
                        cost = cost + gamma * graph.nodes[o]['value'] * prob_other

                # add weight to summed up cost    
                step_cost = graph.get_edge_data(n,m)['weight']
                cost = cost + step_cost
                Q.update({n:cost})

            if Q == {}:
                continue
            best_n = min(Q, key=Q.get)
            best_cost = Q[best_n]
            
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        i += 1
    
    #print(get_node_attributes(graph, 'value'))
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        current_node = init
        has_loop = False
        visited = set()
        while not goal_reached:
            visited.add(current_node)
            desired = graph.nodes[current_node]['next'] # select our desired node
            prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(current_node)))) # get probabilities
            choice = random.random()
            if choice <= prob_success:
                nn = desired # successful transition
            elif choice > prob_success and choice <= prob_success + prob_stay:
                nn = current_node # stay
            else:
                current_range = prob_success + prob_stay
                for o in graph.neighbors(current_node):
                    if o != desired: # make sure that the desired node is not taken into account
                        if choice > current_range and choice <= current_range + prob_other:
                            nn = o
                            break
                        else: current_range += prob_other
            if not has_loop and nn in visited:
                has_loop = True # we don't break here since it is okay to have a loop in the probabilistic case
            path.append(nn)
            current_node = nn
            if nn in goal_region:
                goal_reached = True
    #print("Stages: " + str(i))
    return i, num_actions, path, has_loop

def q_valit_path(graph, init, goal_region, gamma = 1):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    for goal in goal_region:
        set_node_attributes(graph, {goal:0.0}, 'value') # This is the termination action, although it is not an action to speak of.
    
    num_actions = 0
    # main loop
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            best_n = m
            Q = {}
            for n in graph.neighbors(m):
                num_actions += 1
                step_cost = graph.get_edge_data(n,m)['weight']
                cost = gamma * graph.nodes[n]['value'] + step_cost
                Q.update({n:cost})
            
            if Q == {}:
                continue
            best_n = min(Q, key=Q.get)
            best_cost = Q[best_n]

            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        i += 1
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        visited = set()
        current_node = init
        has_loop = False
        while not goal_reached:
            visited.add(current_node)
            nn = graph.nodes[current_node]['next']
            if nn in visited:
                has_loop = True
                break
            path.append(nn)
            current_node = nn
            if nn in goal_region:
                goal_reached = True
    #print("Stages: " + str(i))
    return i, num_actions, path, has_loop

# Below code is taken from:
# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.
# Compute the stationary cost-to-go function and return a solution path.
def valit_path(graph, init, goal_region, gamma = 1):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    for goal in goal_region:
        set_node_attributes(graph, {goal:0.0}, 'value') # This is the termination action, although it is not an action to speak of.
    
    num_actions = 0
    # main loop
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            best_n = m
            for n in graph.neighbors(m):
                num_actions += 1
                step_cost = graph.get_edge_data(n,m)['weight']
                cost = gamma * graph.nodes[n]['value'] + step_cost
                if cost < best_cost:
                    best_cost = cost
                    best_n = n
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        i += 1
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        visited = set()
        current_node = init
        has_loop = False
        while not goal_reached:
            visited.add(current_node)
            nn = graph.nodes[current_node]['next']
            if nn in visited:
                has_loop = True
                break
            path.append(nn)
            current_node = nn
            if nn in goal_region:
                goal_reached = True
    #print("Stages: " + str(i))
    return i, num_actions, path, has_loop