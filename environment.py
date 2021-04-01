import random
import pygame
from car import Car


screen = pygame.display.set_mode((800, 800))
background = pygame.image.load('white.bmp').convert()

class Environment:
    # 100 x 100 map
    def __init__(self):
        self.car = Car(10, 10)
        self.puddle = [[30, 70], [70, 30]] # rectangular puddle ([x1, y1], [x2, y2])
        self.player = pygame.image.load('car.bmp').convert() # change image size
        screen.blit(background, (0, 0))

    def step(self, u):
        if self.car.X[0]>self.puddle[0][0] and self.car.X[0]<self.puddle[0][1] and self.car.X[1]>self.puddle[1][0] and self.car.X[1]<self.puddle[1][1]:
            self.car.step(u, fr=0.2)
        else:
            self.car.step(u, fr=0.7)
    
    def run(self):
        for i in range(5000):
            # display gui
            screen.blit(background, (self.car.X[0], self.car.X[1]))
            # get human control
            u = [0.1, random.random()*0.2-0.1]
            # get NN control
            # apply step
            print(self.car.X)
            self.step(u)
            screen.blit(self.player, (self.car.X[0], self.car.X[1]))
            pygame.display.update()
            pygame.time.delay(100)

env = Environment()
env.run()