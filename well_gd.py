import pygame 
from pygame.locals import *
import numpy as np
pygame.init()

# interesting seeds: 
# 662076453 - Multiple local minima
# 917089864 - Interesting meandering behavior in Momentum

screen = pygame.display.set_mode((1000, 700))

seed = np.random.randint(0, 1000000000)
seed = 770774971
np.random.seed(seed)
print("SEED:", seed)

n = 20
r = 20

num_wells = 4

grid = True

if not grid:
    points = np.random.random((n, 2)) * r   
else:
    #points = np.array([[[i * 2, j * 2] for i in range (j)] for j in range (10)])
    #print(points.shape)
    #points = points.reshape(45, 2)
    points =[ ]
    for j in range (10):
        for i in range (j):
            points.append([i * 2, j * 2])
    
    points = np.array(points)

#points = np.array([[5, 5], [5, 5], [10, 5]]).astype(float)

wells = np.random.random((1, num_wells , 2)) * r
wells = np.repeat(wells, 4, 0)

z = np.zeros((num_wells, 2)).astype(float)
z_2 = np.zeros((num_wells, 2)).astype(float)
gradients = np.zeros((4, num_wells, 2)).astype(float)
momentum = 0.90

colors = [[255, 0, 0], [0, 0, 255], [250, 10, 250], [0, 120, 50]]

cost = []
t = 0 
done = False

def draw_polygon(screen, color, x, y, sides, length, rotate = 0, thickness = 1):
    angle = 3.1415 * 2 / sides
    a = 0 + rotate * 3.1415/180
    positions = [] 
    for i in range (sides):
        posx = x + np.sin(a) * length
        posy = y + np.cos(a) * length 
        a += angle

        positions.append([int(posx), int(posy)])
    
    pygame.draw.polygon(screen, color, positions, thickness)

while not done:
    t += 1
    screen.fill((255, 255, 255))
    gradients = np.zeros((4, num_wells, 2)).astype(float)
    points_belong = np.zeros((num_wells,1)) + 0.001

    error = np.zeros((4))

    for p in points:
        for experiment in range (len(wells)):
            distance = np.sqrt(np.sum((wells[experiment] - p) ** 2, axis = 1))      # Distance to each well
            idx = np.argmin(distance)

            error[experiment] += np.sqrt(np.min(distance))

            #10 x 10 x 2
            #print(np.sum((2 * (points - wells[idx]) * -1), axis = 0).shape)
            if experiment < 2:
                shared_term = (np.sum((p - wells[experiment][idx]) ** 2) + 0.001) ** -0.5
                gradients[experiment][idx] += 0.5 * shared_term * (2 * (p - wells[experiment][idx]) * -1)

            elif experiment == 2:
                shared_term = (np.sum((p - wells[experiment][idx]) ** 2) + 0.001) ** -0.5

                gradients[experiment][idx] += 0.5 * shared_term * (2 * (p - wells[experiment][idx]) * -1)
                gradients[experiment] += 0.5 * (np.sum((p - wells[experiment]) ** 2, axis = -1, keepdims = True) + 0.001) ** -0.5 * (2 * (p - wells[experiment]) * -1) * 0.02

            elif experiment == 3:       # iterative method, mallen et al.
                gradients[experiment][idx] += p
                points_belong[idx] += 1

    wells[0] -= gradients[0] * 0.01
    
    if t % 1 == 0:
        # Momentum for experiment 1
        z = z * momentum + gradients[1]
        wells[1] -= z * 0.002
    
        # Momentum for experiment 1
        z_2 = z_2 * momentum + gradients[2]
        wells[2] -= z_2 * 0.002

    if t % 50 == 0:
        #alpha = 0.8
        #wells[3] = wells[3] * alpha + (1 - alpha) * gradients[3] / points_belong      # move to center of mass
        wells[3] = gradients[3] / points_belong

    if t % 1000 == 0:
        seed = np.random.randint(0, 1000000000)
        np.random.seed(seed)
        print("SEED:", seed)
        
        wells = np.random.random((1, num_wells, 2)) * r
        wells = np.repeat(wells, 4, 0)
        if not grid:
            points = np.random.random((n, 2)) * r
        else:
            points = np.array([[[i * 4, j * 4] for i in range (6)] for j in range (6)])
            points = points.reshape(36, 2)
            grid = False

        z = np.zeros((num_wells, 2)).astype(float)
        z_2 = np.zeros((num_wells, 2)).astype(float)

        cost = []

    
    if t % 5 == 0:
        cost.append(error)
    #if (len(cost) > 200):
    #    del cost[0]
    

    c_idx = 0

    if len(cost) > 1:
        # likely largest error, so get the maximum and scale down
        m = np.max(cost[0])
        scale = 200 / m
    else:
        scale = 1

    prev_c = 0
    for c in range (len(cost) - 1):
        
        if len(cost) > 200:
            if c % (len(cost) / 200) < 1:
                for e in range(len(cost[c])):
                    thickness = e == np.argmin(cost[c])
                    pygame.draw.line(screen, colors[e], (500 + 50 + c_idx * 2, 700 - (cost[prev_c][e] * scale)), (500 + 50 + c_idx * 2 + 2, 700 - (cost[c][e]* scale)), thickness + 1)
                prev_c = c
                c_idx += 1
        else:
            for e in range(len(cost[c])):
                thickness = e == np.argmin(cost[c])
                pygame.draw.line(screen, colors[e], (500 + 50 + c * 2, 700 - (cost[c][e]* scale)), (500 + 50 + c  * 2 + 2, 700 - (cost[c + 1][e]* scale)), thickness + 1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            pygame.quit()

    zoom = 15

    for p in points:
        pygame.draw.circle(screen, (0, 0, 0), (p * zoom + 100).astype(int), 5, 1)
    
    for i, well_set in enumerate(wells):
        for w in well_set:
            if i == 3:
                draw_polygon(screen, colors[i], w[0] * zoom + 100 , w[1] * zoom + 100, 4, 8, rotate = 45)
                #pygame.draw.rect(screen, colors[i], (w[0] * zoom + 100 - 3, w[1] * zoom + 100 - 3, 6, 6), 1)
            elif i == 0:
                pygame.draw.line(screen, colors[i], w * zoom + 100 - 6, w * zoom + 100 + 6, 1)
                pygame.draw.line(screen, colors[i], (w[0] * zoom + 100 - 6, w[1] * zoom + 100 + 6), (w[0] * zoom + 100 + 6, w[1] * zoom + 100 - 6), 2)
            elif i == 1:
                draw_polygon(screen, colors[i], w[0] * zoom + 100 , w[1] * zoom + 100, 4, 8, rotate = 0, thickness = 2)
            elif i == 2:
                draw_polygon(screen, colors[i], w[0] * zoom + 100 , w[1] * zoom + 100, 3, 8, rotate = 180)

    pygame.display.flip()



