# Modified from:
# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

from math import sqrt
import pygame

white = 255, 255, 255
grey = 100, 100, 100

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

def draw_graph_edges(g,screen):
    for i,j in g.edges:
        pygame.draw.line(screen,white,g.nodes[i]['point'],g.nodes[j]['point'],2)

def draw_discs(dlist,screen):
    for d in dlist:
        pygame.draw.circle(screen,grey,[d[0],d[1]],d[2])

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