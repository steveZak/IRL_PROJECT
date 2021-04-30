import random
import pygame
from pygame.locals import *
import numpy as np
import math
from car import Car

def rot_center(image, angle, x, y):
    rotated_image = pygame.transform.rotate(image, angle*180/np.pi)
    new_rect = rotated_image.get_rect(center = image.get_rect(center = (x, y)).center)
    return rotated_image, new_rect

def blitRotate(surf, image, pos, originPos, angle):
# def blitRotate(surf, image, originPos, angle):

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
    # origin = (originPos[0] + min_box[0] - pivot_move[0], - originPos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)

    # rotate and blit the image
    surf.blit(rotated_image, origin)
  
    # draw rectangle around the image
    pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()),2)

class Environment:
    # 100 x 100 map
    def __init__(self, gui=True):
        pygame.init()
        if gui:
            self.screen = pygame.display.set_mode((1000, 1000))
            self.icon = pygame.image.load('_car.bmp')
            self.icon = self.icon.convert() # now you can convert
            # clock = pygame.time.Clock()
            w, h = self.icon.get_size()
            pivot= [w/2, h/2]
        self.clock = pygame.time.Clock()
        # self.dt = self.clock.get_time() / 1000
        self.dt = 0.05
        # print(self.clock.get_time())
        self.Q = np.array([[-1, 0, 0, 0, 0, 0],[0, -1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, -1, 0, 0],[0, 0, 0, 0, -1, 0],[0, 0, 0, 0, 0, 0]]) 
    
    def step(self, u, gui = True):
        if self.icepatch is not None and self.car.X[0] > self.icepatch[0][0] and self.car.X[0] < self.icepatch[0][1] and self.car.X[1] > self.icepatch[1][0] and self.car.X[1] < self.icepatch[1][1]:
            self.car.step(u, self.dt, skid=True)
        else:
            self.car.step(u, self.dt, skid=False)
        if self.blowout is not None and self.car.X[0] > self.blowout[0][0] and self.car.X[0] < self.blowout[0][1] and self.car.X[1] > self.blowout[1][0] and self.car.X[1] < self.blowout[1][1]:
            self.car.step(u, self.dt, blowout=True)
        else:
            self.car.step(u, self.dt, blowout=False)
        if gui:
            white = [255, 255, 255]
            self.screen.fill(white)
            # pygame.event.pump()
            # icepatch
            if self.icepatch is not None:
                pygame.draw.rect(self.screen, [0, 0, 255], (self.icepatch[0][0], self.icepatch[1][0], self.icepatch[0][1]-self.icepatch[0][0], self.icepatch[1][1]-self.icepatch[1][0]))
            if self.blowout is not None:
                pygame.draw.rect(self.screen, [0, 255, 0], (self.blowout[0][0], self.blowout[1][0], self.blowout[0][1]-self.blowout[0][0], self.blowout[1][1]-self.blowout[1][0]))
            # purple goal
            pygame.draw.circle(self.screen, [255, 0, 255], (self.goal[0], self.goal[1]), 20)
            # red start
            pygame.draw.circle(self.screen, [255, 0, 0], (self.start[0], self.start[1]), 20)
            # apply step
            # print(self.car.X)
            x = self.car.X[0]
            y = self.car.X[1]
            rotated = pygame.transform.rotate(self.icon, self.car.angle)
            rect = rotated.get_rect()
            self.screen.blit(rotated, self.car.position - (rect.width / 2, rect.height / 2))
            pygame.display.flip()
            self.clock.tick(60)
        reward = np.matmul(self.car.X - self.goal, np.matmul(self.Q, self.car.X - self.goal))
        return reward

    def reset(self, noise=True, X=None, goal=None, icepatch=None, blowout=None):
        if noise:
            if icepatch is None:
                self.icepatch = [[300+(random.random()*100), 500+random.random()*100], [300+(random.random()*100), 500+random.random()*100]]
            else:
                self.icepatch = icepatch
            if blowout is None:
                x = random.random()*900
                y = random.random()*900
                self.blowout = [[x, y], [(random.random()*200), random.random()*200]]
                # self.blowout = [[300+(random.random()*100), 500+random.random()*100], [300+(random.random()*100), 500+random.random()*100]]
            else:
                self.blowout = blowout
        else:
            self.icepatch = None
            self.blowout = None
        if X is None:
            x = 100+800*random.random()
            y = 100+800*random.random()
            x_g = 100+800*random.random()
            y_g = 100+800*random.random()
            tht = -(np.arctan2([y_g-y], [x_g-x])[0]+random.random()*np.pi/5-np.pi/10)*180/np.pi
            self.car = Car([x, y, tht, 0, 0, 0])
            self.start = np.array([x, y, tht, 0, 0, 0])
            self.goal = np.array([x_g, y_g, tht, 0, 0, 0])
        else:
            self.car = Car(X)
            self.start = X
            self.goal = goal
        return self.goal, self.car.X, self.icepatch, self.blowout
