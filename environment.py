import random
import pygame
from pygame.locals import *
import numpy as np
import math
from car import Car

pygame.init()
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
icon = pygame.image.load('_car.bmp')
w, h = icon.get_size()
pivot= [w/2, h/2]
pos = [0, 1000]


def rot_center(image, angle, x, y):
    rotated_image = pygame.transform.rotate(image, angle*180/np.pi)
    new_rect = rotated_image.get_rect(center = image.get_rect(center = (x, y)).center)
    return rotated_image, new_rect

def blitRotate(surf, image, pos, originPos, angle):

    # calcaulate the axis aligned bounding box of the rotated image
    w, h       = image.get_size()
    box        = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box    = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box    = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # calculate the translation of the pivot 
    pivot        = pygame.math.Vector2(originPos[0], -originPos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move   = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - originPos[0] + min_box[0] - pivot_move[0], pos[1] - originPos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)

    # rotate and blit the image
    surf.blit(rotated_image, origin)
  
    # draw rectangle around the image
    pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()),2)

def getHumanControl():
    yaw = 0
    thrust = -20
    # print(pygame.K_LEFT)
    # print(pygame.key.get_pressed()[pygame.K_LEFT])
    keys = pygame.key.get_pressed()
    print(keys[K_LEFT])
    if keys[K_LEFT]:
        yaw = 0.0001
    if keys[K_RIGHT]:
        yaw = -0.0001
    if keys[K_UP]:
        thrust = -40
    if keys[K_DOWN]:
        thrust = -40
    print(thrust)
    # for event in pygame.event.get():
    #     # print(event)
    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_LEFT:
    #             yaw = 0.0001
    #         elif event.key == pygame.K_RIGHT:
    #             yaw =-0.0001
    #         if event.key == pygame.K_UP:
    #             thrust = -40
    #         elif event.key == pygame.K_DOWN:
    #             thrust = 40
    # print(thrust)
    return thrust, yaw

class Environment:
    # 100 x 100 map
    def __init__(self, gui=True):
        if gui:
            self.icon = icon.convert() # now you can convert
        self.Q = np.array([[-1, 0, 0, 0, 0, 0],[0, -1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, -1, 0, 0],[0, 0, 0, 0, -1, 0],[0, 0, 0, 0, 0, 0]]) 
    
    def step(self, u, gui = True):
        if self.puddle is not None and self.car.X[0] > self.puddle[0][0] and self.car.X[0] < self.puddle[0][1] and self.car.X[1] > self.puddle[1][0] and self.car.X[1] < self.puddle[1][1]:
            self.car.step(u, fr=0.2)
            # print("in a puddle")
        else:
            self.car.step(u, fr=0.7)
        if gui:
            white = [255, 255, 255]
            screen.fill(white)
            positions = list()
            screen.fill(white)
            # pygame.draw.circle(screen, [255, 0, 0], (200, 800), 20)
            # blue goal
            pygame.draw.circle(screen, [0, 0, 255], (self.goal[0], self.goal[1]), 20)
            # red start
            pygame.draw.circle(screen, [255, 0, 0], (200, 800), 20)
            # u = [1, random.random()*2] # 0.1/0.2-0.1
            # u = getHumanControl()
            pygame.event.pump()
            # get NN control
            # apply step
            x = self.car.X[0]# reversed
            y = self.car.X[1]
            sin_a, cos_a = math.sin(self.car.X[2]), math.cos(self.car.X[2])
            min_x, min_y = min([0, sin_a*h, cos_a*w, sin_a*h + cos_a*w]), max([0, sin_a*w, -cos_a*h, sin_a*w - cos_a*h])
            # origin = (pos[0] - pivot[0] + min_x - x, pos[1] - pivot[1] - min_y + y)
            # _ray = (pos[0] + min_x - x, pos[1] - min_y + y)
            origin = (pivot[0] - min_x + x, pivot[1] - min_y + y)
            _ray = (min_x + x, min_y + y)
            rotated_image = pygame.transform.rotate(icon, -self.car.X[2]*180/np.pi)
            screen.blit(rotated_image, origin)
            positions.append(_ray)
            pygame.display.update()
            pygame.time.delay(1)
            pygame.display.flip()
        reward = np.matmul(self.car.X - self.goal, np.matmul(self.Q, self.car.X - self.goal))
        return reward
    
    def reset(self, noise=True, X=[200, 800, -np.pi/4, 0.000001, 0, 0], puddle=None):
        if noise:
            if puddle is not None:
                self.puddle = [[random.random()*50, 50+random.random()*50], [50+random.random()*50, random.random()*50]]
            else:
                self.puddle = puddle
        else:
            self.puddle = None
        self.car = Car(X)
        self.goal = [800, 200, np.pi/4, 0, 0, 0]
        return self.goal, X, puddle

    def run(self, gui=True):
        white = [255, 255, 255]
        screen.fill(white)
        positions = list()
        for i in range(5000):
            # get human control
            u = [-10, 0.0001]
            self.step(u)
            if gui:
                screen.fill(white)
                pygame.draw.circle(screen, [255, 0, 0], (200, 800), 20)
                pygame.draw.circle(screen, [0, 0, 255], (800, 200), 20)
                # u = [1, random.random()*2] # 0.1/0.2-0.1
                # u = getHumanControl()
                pygame.event.pump()
                # get NN control
                # apply step
                x = self.car.X[0]
                y = self.car.X[1]
                sin_a, cos_a = math.sin(self.car.X[2]), math.cos(self.car.X[2])
                min_x, min_y = min([0, sin_a*h, cos_a*w, sin_a*h + cos_a*w]), max([0, sin_a*w, -cos_a*h, sin_a*w - cos_a*h])
                # origin = (pos[0] - pivot[0] + min_x - x, pos[1] - pivot[1] - min_y + y)
                # _ray = (pos[0] + min_x - x, pos[1] - min_y + y)
                origin = (pivot[0] - min_x + x, pivot[1] - min_y + y)
                _ray = (min_x + x, min_y + y)
                rotated_image = pygame.transform.rotate(icon, self.car.X[2]*180/np.pi)
                screen.blit(rotated_image, origin)
                positions.append(_ray)
                pygame.display.update()
                pygame.time.delay(1)
                pygame.display.flip()

# env = Environment()
# env.reset()
# env.run()