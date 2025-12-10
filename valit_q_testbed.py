#! /usr/bin/env python3

# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

from networkx.classes.function import get_node_attributes
import pygame, time
from pygame.locals import *
import networkx as nx
from tkinter import *
from ast import literal_eval
from rppl_util_necessary import *
from q_learning_functions import *
from valit_functions import *


dims = 20 # number of samples per axis
radius = 1 # neightborhood radius (1 = four-neighbors)
exnum = 2 # example number
xmax = 800 # force a square environment
ymax = 800

black = 0, 0, 0
red = 255, 0, 0
blue = 50, 50, 255
green = 0, 255, 0

screen = pygame.display.set_mode([xmax,ymax])
use_qlearning = False
pygame.display.set_caption('Grid Planner')

# This corresponds to GUI button 'Draw' that runs the example.
def Draw():
    obstacles = literal_eval(problines[exnum*3])
    initial = literal_eval(problines[exnum*3+1])
    goal = literal_eval(problines[exnum*3+2])

    global G
    arr = [[0 for i in range(dims)] for j in range(dims)]
    actions = generate_neighborhood_indices(radius)
    G = nx.Graph()
    pygame.init()
    screen.fill(black)
    incrementy = 0
    i = 0
    length = 0
    
    # construct grid
    for y in range(dims):
        if y > 0:
            incrementy += ymax/dims + (ymax/dims)/(dims-1)
        incrementx = 0
        for x in range(dims):
            G.add_node(i, point=(incrementx,incrementy))
            incrementx += xmax/dims + (xmax/dims)/(dims-1)
            arr[y][x] = i
            i += 1
    for x in range(dims):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                quit()
        for y in range(dims):
            for u in actions:
                if (0 <= x + u[0] <= dims-1 and 0 <= y + u[1] <= dims-1 
                    and safe(G.nodes[arr[y][x]]['point'],G.nodes[arr[y+u[1]][x+u[0]]]['point'],obstacles) 
                    and not G.has_edge(arr[y][x],arr[y+u[1]][x+u[0]])
                    ):
                    G.add_edge(arr[y][x],arr[y+u[1]][x+u[0]],weight=dist2(G.nodes[arr[y][x]]['point'],G.nodes[arr[y+u[1]][x+u[0]]]['point']))
    # The next three lines delete the obstacle nodes (optional).
    #for i in range(len(G.nodes)):
    #        if point_inside_discs(G.nodes[i]['point'],obstacles):
    #            G.remove_node(i)
    draw_discs(obstacles, screen)
    draw_graph_edges(G, screen)
    p1index = find_closest_node(initial,G.nodes)
    p2index = find_closest_node(goal,G.nodes)
    # Print edge cost/weight
    # for (u,v,c) in G.edges().data():
    #     print("Edge (" + str(u) + ", " + str(v) +"): " + str(c))

    # Use a radius parameter to find the neighbors that will define the goal region
    goal_radius = 0
    goal_indices = list(nx.single_source_shortest_path_length(G, p2index, cutoff=goal_radius).keys())

    #Since the graph is undirected, this is equivalent to checking if there is a path from p1index to any of the goal_indices
    if nx.has_path(G,p1index,p2index):
        t = time.time()
        if use_qlearning:
            path = q_learning_dc_path(G, p1index, goal_indices)
            print('Q-learning:   time elapsed:     ' + str(time.time() - t) + ' seconds')
        # elif use_dijkstra:
        #     path = nx.dijkstra_path(G,p1index,p2index)
        #     print('dijkstra:    time elapsed:     ' + str(time.time() - t) + ' seconds')
        else:
            path = random_valit_path(G,p1index,goal_indices, True)
            print('value iteration: time elapsed: ' + str(time.time() - t) + ' seconds')
        print("Shortest path: " + str(len(path)))
        for l in range(len(path)):
            if l > 0:
                pygame.draw.line(screen,green,G.nodes[path[l]]['point'],G.nodes[path[l-1]]['point'],5)
                if G.get_edge_data(path[l],path[l-1]) is not None: # When there are loops, there is no weight in some cases
                    length += G.get_edge_data(path[l],path[l-1])['weight']
        pygame.display.set_caption('Grid Planner, Euclidean Distance: ' +str(length))
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
    exnum = int(val) - 1

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

exscale = Scale(m1, orient = HORIZONTAL, from_=1, to=num_of_ex, resolution=1, command=SetExNum, label='Example Number')
exscale.set(int(exnum))
m1.add(exscale)

master.mainloop()