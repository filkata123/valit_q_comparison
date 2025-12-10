# Modified from:
# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

from math import sqrt
import networkx as nx

def sqr(a):
    return a * a

def dist2(p, q):
    a = sqrt(sqr(p[0] - q[0]) + sqr(p[1] - q[1]))
    return a

def vlen(v):
    return sqrt(v[0]*v[0]+v[1]*v[1])

def detect( A, B, C, r ):#https://stackoverflow.com/questions/26725842/how-to-pick-up-line-segments-which-is-inside-or-intersect-a-circle
    AB = (B[0] - A[0], B[1]-A[1])
    AC = (C[0] - A[0], C[1]-A[1])
    BC = (C[0]-B[0], C[1]-B[1])

    if vlen(BC) < r or vlen(AC) < r: return True
    
    abl = vlen(AB)
    AB_normalized = (AB[0] / abl , AB[1] / abl)
    AP_distance = AC[0] * AB_normalized[0]  +  AC[1] * AB_normalized[1]
    AP = (AP_distance * AB_normalized[0], AP_distance * AB_normalized[1])
    
    AP_proportion = AP_distance / vlen( AB )   
    in_segment =   0 <= AP_proportion <= 1

    CP = (AP[0] - AC[0], AP[1]-AC[1])
    in_circle = vlen( CP ) < r

    return in_circle and in_segment


def find_closest_node(mpos,nodes):
    a = [dist2(mpos, nodes[0]['point']),0]
    for i in nodes:
        if i > 0:
            b = [dist2(mpos, nodes[i]['point']),i]
            if a[0] > b[0]:
                a = [dist2(mpos, nodes[i]['point']),i]
    return a[1]

def safe(a,b,obsts):
    for i in range(len(obsts)):
        if detect(a,b,(obsts[i][0],obsts[i][1]),obsts[i][2]):
            return False
    return True

def generate_neighborhood_indices(radius):
    neighbors = []
    k = int(radius+2)
    for i in range(-k,k):
        for j in range(-k,k):
            if 0 < vlen([i,j]) <= radius:
                neighbors.append([i,j])
    return neighbors

def build_grid_graph(dims, radius, obstacles, xmax=800, ymax=800):
    arr = [[0 for _ in range(dims)] for _ in range(dims)]
    actions = generate_neighborhood_indices(radius)
    G = nx.Graph()
    incrementy = 0
    i = 0

    # place nodes in a grid (points in continuous space)
    for y in range(dims):
        if y > 0:
            incrementy += ymax/dims + (ymax/dims)/(dims-1)
        incrementx = 0
        for x in range(dims):
            G.add_node(i, point=(incrementx, incrementy))
            incrementx += xmax/dims + (xmax/dims)/(dims-1)
            arr[y][x] = i
            i += 1
    # add edges if line-segment between points is safe w.r.t obstacles
    for x in range(dims):
        for y in range(dims):
            for u in actions:
                nx_ = x + u[0]
                ny_ = y + u[1]
                if 0 <= nx_ <= dims-1 and 0 <= ny_ <= dims-1:
                    a = arr[y][x]
                    b = arr[ny_][nx_]
                    if safe(G.nodes[a]['point'], G.nodes[b]['point'], obstacles) and not G.has_edge(a, b):
                        G.add_edge(a, b, weight=dist2(G.nodes[a]['point'], G.nodes[b]['point']))
    return G