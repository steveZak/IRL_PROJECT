import random
import pygame
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
    print(pygame.K_LEFT)
    print(pygame.key.get_pressed()[pygame.K_LEFT])
    if pygame.key.get_pressed()[pygame.K_LEFT]:
        yaw = 0.0001
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
        yaw = -0.0001
    if pygame.key.get_pressed()[pygame.K_UP]:
        thrust = -40
    if pygame.key.get_pressed()[pygame.K_DOWN]:
        thrust = -40
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
    def __init__(self):
        self.car = Car(-200, -200)
        self.puddle = [[30, 70], [70, 30]] # rectangular puddle ([x1, y1], [x2, y2])
        
        # screen = pygame.display.set_mode(icon.get_rect().size, 0, 32)
        self.icon = icon.convert() # now you can convert 
        # screen.blit(background, (0, 0))

    def step(self, u):
        if self.car.X[0]>self.puddle[0][0] and self.car.X[0]<self.puddle[0][1] and self.car.X[1]>self.puddle[1][0] and self.car.X[1]<self.puddle[1][1]:
            self.car.step(u, fr=0.2)
        else:
            self.car.step(u, fr=0.7)
    
    def run(self):
        white = [255, 255, 255]
        screen.fill(white)
        positions = list()
        for i in range(5000):
            # display gui
            # get human control
            screen.fill(white)
            pygame.draw.circle(screen, [255, 0, 0], (200, 800), 20)
            pygame.draw.circle(screen, [0, 0, 255], (800, 200), 20)
            # u = [1, random.random()*2] # 0.1/0.2-0.1
            # u = [-10, 0.0001]
            u = getHumanControl()
            pygame.event.pump()
            # get NN control
            # apply step
            self.step(u)
            print(self.car.X)
            # _pos = np.multiply([[math.cos(self.car.X[2]), -math.sin(self.car.X[2])],[math.sin(self.car.X[2]), math.cos(self.car.X[2])]], [self.car.X[0],self.car.X[1]])
            # x = _pos[0][0]
            # y = _pos[1][1]
            x = self.car.X[0]
            y = self.car.X[1]
            sin_a, cos_a = math.sin(self.car.X[2]), math.cos(self.car.X[2])
            min_x, min_y = min([0, sin_a*h, cos_a*w, sin_a*h + cos_a*w]), max([0, sin_a*w, -cos_a*h, sin_a*w - cos_a*h])
            origin = (pos[0] - pivot[0] + min_x - x, pos[1] - pivot[1] - min_y + y)
            _ray = (pos[0] + min_x - x, pos[1] - min_y + y)
            rotated_image = pygame.transform.rotate(icon, self.car.X[2]*180/np.pi)
            screen.blit(rotated_image, origin)
            positions.append(_ray)
            # if i>300:
            #     print('ay')
            #     screen.fill([255, 0, 0], (positions[i-300], (2, 2)))
            # screen.blit(rot_center(self.icon, self.car.X[2], self.car.X[0], self.car.X[1])[0], (self.car.X[0], self.car.X[1]))
            # rotated_image = pygame.transform.rotate(self.icon, i)
            # screen.blit(rotated_image, (5,5))
            # pygame.display.flip()
            pygame.display.update()
            pygame.time.delay(1)
            pygame.display.flip()

env = Environment()
env.run()