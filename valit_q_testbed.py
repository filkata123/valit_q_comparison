#! /usr/bin/env python3
# Modified from:
# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

from networkx.classes.function import get_node_attributes
import pygame
from pygame.locals import *
from tkinter import *
from valit_q_testbed_helper import init_problem, find_path
from q_learning_functions import *
from valit_functions import *
from dijkstra_functions import *

dims = 20 # number of samples per axis
radius = 1 # neightborhood radius (1 = four-neighbors)
exnum = 2 # example number

use_qlearning = False

white = 255, 255, 255
grey = 100, 100, 100
black = 0, 0, 0
red = 255, 0, 0
blue = 50, 50, 255
green = 0, 255, 0

def draw_graph_edges(g,screen):
    for i,j in g.edges:
        pygame.draw.line(screen,white,g.nodes[i]['point'],g.nodes[j]['point'],2)

def draw_discs(dlist,screen):
    for d in dlist:
        pygame.draw.circle(screen,grey,[d[0],d[1]],d[2])

def draw_pygame(graph, obstacles, p1index, has_path, path, euclidean_distance, goal_indices):
    xmax = 800 # force a square environment
    ymax = 800
    screen = pygame.display.set_mode([xmax,ymax])
    pygame.display.set_caption('Grid Planner')
    pygame.init()
    screen.fill(black)
    draw_discs(obstacles, screen)
    draw_graph_edges(graph, screen)

    if has_path:
        for l in range(len(path)):
            if l > 0:
                pygame.draw.line(screen,green,G.nodes[path[l]]['point'],G.nodes[path[l-1]]['point'],5)
        pygame.display.set_caption('Grid Planner, Euclidean Distance: ' +str(euclidean_distance))
    else:
        print('Path not found')
        pygame.display.set_caption('Grid Planner')
    pygame.draw.circle(screen,green,G.nodes[p1index]['point'],10)
    for g in goal_indices:
        pygame.draw.circle(screen,red,G.nodes[g]['point'],10)
    # Old implementation of visualization
    #pygame.draw.circle(screen,green,initial,10)
    #pygame.draw.circle(screen,red,goal,10)
    pygame.display.update()
    #pygame.image.save(screen, "screenshot.png")

def draw_visits_heatmap(graph, obstacles, visits, p1index, goal_indices, title="State-Action Visit Frequency Heatmap"):
    xmax = 800
    ymax = 800
    screen = pygame.display.set_mode([xmax, ymax])
    pygame.display.set_caption(title)
    screen.fill(black)
    
    # Draw obstacles
    draw_discs(obstacles, screen)
    
    edge_visits = []
    for i, j in graph.edges:
        edge_visits.append(visits.get((i, j), 0) + visits.get((j, i), 0))

    min_visits = min(v for v in edge_visits)
    max_visits = max(edge_visits)
    print(f"Min visits: {min_visits}")
    print(f"Max visits: {max_visits}")

    visit_range = max_visits - min_visits if max_visits > min_visits else 1
    
    # Draw edges with color based on visit frequency
    for i, j in graph.edges:
        # Check both (i,j) and (j,i) for undirected graph
        visit_count = visits.get((i, j), 0) + visits.get((j, i), 0)
        
        if visit_count == 0:
            # Unvisited edges in dark gray
            edge_color = (50, 50, 50)
            thickness = 1
        else:
            # Normalize to [0, 1]
            normalized = 1 - math.exp(-visit_count / max_visits)
            
            if normalized < 0.5:
                # Dark blue to cyan
                t = normalized * 2  # [0, 1]
                r = 0
                g = int(180 * t)          # 0 -> 180
                b = int(100 + 155 * t)    # 100 -> 255
            else:
                # Cyan to white
                t = (normalized - 0.5) * 2  # [0, 1]
                r = int(255 * t)           # 0 -> 255
                g = int(180 + 75 * t)      # 180 -> 255
                b = 255
                
            edge_color = (r, g, b)
            # Thickness scales with visits (1 to 5 pixels)
            thickness = 1 + int(6 * math.sqrt(normalized))
        pygame.draw.line(screen, edge_color, graph.nodes[i]['point'], graph.nodes[j]['point'], thickness)

    pygame.draw.circle(screen,green,G.nodes[p1index]['point'],10)
    for g in goal_indices:
        pygame.draw.circle(screen,red,G.nodes[g]['point'],10)
    
    pygame.display.update()

# This corresponds to GUI button 'Draw' that runs the example.
def Draw():
    global G
    G, p1index, p2index, obstacles, goal_indices = init_problem(problines, exnum, dims, radius)
    visits = {}
    viz_visits = False
    if use_qlearning:
        has_path, path, goal_in_path, euclidean_distance, elapsed_time, path_length, num_iterations_or_episodes, num_actions, has_loop, converged_at_action, visits = find_path(G, p1index,p2index, q_learning_stochastic_path, (G, p1index, goal_indices, 1000, 500, 0.999, 1, 0.9, True, 0.9))
        if converged_at_action != 0:
            print("Converged at " + str(converged_at_action))
        else:
            print("No convergence.")
        print('Q-learning:   time elapsed:     ' + str(elapsed_time) + ' seconds')
        print("Number of episodes: " + str(num_iterations_or_episodes))
    else:
        has_path, path, goal_in_path, euclidean_distance, elapsed_time, path_length, num_iterations_or_episodes, num_actions, has_loop, converged_at_action, visits = find_path(G, p1index,p2index, valit_path, (G, p1index, goal_indices))
        print('value iteration:   time elapsed:     ' + str(elapsed_time) + ' seconds')
        print("Number of iterations: " + str(num_iterations_or_episodes))
    if goal_in_path:
        print("Goal reached!")
    if has_loop:
        print("Loop encountered.")
    print("Number of actions taken overall: " + str(num_actions))
    print("Shortest path found: " + str(path_length))
    if viz_visits:
        draw_visits_heatmap(G, obstacles, visits,p1index,goal_indices, title=f"Q-Learning Visits (Example {exnum})")
    else:
        draw_pygame(G, obstacles, p1index, has_path, path, euclidean_distance, goal_indices)
        
# get example list
problem = open('problem_circles.txt')
problines = problem.readlines()
problem.close()
num_of_ex = len(problines)/3
# The rest is for the GUI.

def SwitchType():
    global use_qlearning
    if use_qlearning:
        use_qlearning = False
        print("Switched to Value Iteration")
    else:
        use_qlearning = True
        print("Switched to Q-learning")
    Draw()

def SetDims(val):
    global dims
    dims = int(val)

def SetRadius(val):
    global radius
    radius = float(val)

def SetExNum(val):
    global exnum
    exnum = int(val)

def Exit():
    master.destroy()

def SaveData():
    data = open('valit_data.txt','w')
    data.write(str(get_node_attributes(G, 'value')) + '\n' + str(get_node_attributes(G, 'point')) + '\n' + str(dims))
    data.close()

master = Tk()
master.title('Grid-Planner GUI')
master.geometry("630x80")

m1 = PanedWindow(master,borderwidth=10,bg="#000000")
m1.pack(fill = BOTH,expand = 1)

exitbutton = Button(m1, text='     Quit     ',command=Exit,fg='red')
m1.add(exitbutton)

savebutton = Button(m1, text='     Save     ',command=SaveData,fg='blue')
m1.add(savebutton)

switchbutton = Button(m1, text='   Change   \n   Planner   ',command=SwitchType,fg='brown')
m1.add(switchbutton)

drawbutton = Button(m1, text='     Draw     ',command=Draw,fg='green')
m1.add(drawbutton)

dimsscale = Scale(m1, orient = HORIZONTAL, from_=2, to=200, resolution=1, command=SetDims, label='Resolution (n * n)')
dimsscale.set(dims)
m1.add(dimsscale)

radscale = Scale(m1, orient = HORIZONTAL, from_=1, to=10, resolution=0.1, command=SetRadius, label='Neighbor Radius')
radscale.set(radius)
m1.add(radscale)

exscale = Scale(m1, orient = HORIZONTAL, from_=0, to=num_of_ex-1, resolution=1, command=SetExNum, label='Example Number')
exscale.set(int(exnum))
m1.add(exscale)

master.mainloop()