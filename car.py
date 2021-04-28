# import math
# import numpy as np

# class Car:
#     def __init__(self, X):
#         # toyota corolla
#         self.fr = 0.9 # friction coefficient?
#         self.m = 1066
#         self.Caf = 0.1
#         self.Csf = 0.1
#         self.Car = 0.1
#         self.Iz = 1706
#         self.lf = 1.054
#         self.lr = 2.372-1.054
#         self.beta_x = 0 # road slope angle
#         self.beta_y = 0 # road tilt angle
#         self.reff = 1*0.264 # estimated wheel radius
#         self.X = X # x, y, tht, x_dot, y_dot, tht_dot
    
#     def step(self, u, delta_t=0.1, fr=0.9):
#         # page 38 in https://core.ac.uk/download/pdf/81577667.pdf with air resistance set to 0
#         # X_dot = Ax+Bu
#         self.fr = fr
#         # X_dot = np.add(
#         #          [self.X[3],\
#         #          self.X[4],\
#         #          self.X[5],\
#         #          -self.X[4]*self.X[5]-np.sign(self.X[3])*(0.5*0*0*0*(self.X[4]**2)+self.fr*self.m*9.81+self.m*9.81*np.sin(self.beta_x))/self.m,\
#         #          -self.X[3]*self.X[5]+(2*self.Car*(self.lr*self.X[5]-self.X[4])/self.X[3]-9.81*np.sign(self.beta_y))/self.m,\
#         #          self.lr*(2*self.Car*(self.lr*self.X[5]-self.X[4])/(self.X[3]*self.m))/self.Iz]\
#         #       , [
#         #          0,
#         #          0,
#         #          0,
#         #          (self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0]))*np.cos(u[1])-2*self.Caf*(u[1]+(self.lf*self.X[5]-self.X[4]/self.X[3]))*np.sin(u[1])+self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0])))/self.m,
#         #          (self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0]))*np.sin(u[1])-2*self.Caf*(u[1]+(self.lf*self.X[5]-self.X[4]/self.X[3]))*np.cos(u[1]))/self.m,
#         #          self.lf*(self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0]))*np.sin(u[1])-2*self.Caf*(u[1]+(self.lf*self.X[5]-self.X[4]/self.X[3]))*np.cos(u[1]))/self.Iz,
#         #         ])
#         # X_dot = [self.X[3]+u[0]*self.reff*math.cos(u[1]),\
#         #          self.X[4]+u[0]*self.reff*math.sin(u[1]),\
#         #          self.X[5]+(self.X[4]+u[0]*self.reff*math.sin(u[1]))/(self.lr+self.lf),\
#         #          0,\
#         #          0,\
#         #          0]
#         tau = u[0]
#         delta = u[1]
#         # print((tau, delta))
#         # print(self.X)
#         if np.abs(tau/self.reff)<self.m*9.81*self.fr:
#             __x = tau/(self.m*self.reff)*math.cos(self.X[2])
#             __y = tau/(self.m*self.reff)*math.sin(self.X[2]) #delta
#         else:
#             __x = 9.81*self.fr*math.cos(self.X[2]) # *m?
#             __y = 9.81*self.fr*math.sin(self.X[2])
#         # (__x, __y) = np.multiply([[math.cos(self.X[2]), -math.sin(self.X[2])],[math.sin(self.X[2]), math.cos(self.X[2])]], [__x,__y])
#         # __x = __x[0]
#         # __y = __y[0]
#         _x = self.X[3]
#         _y = self.X[4]
#         _tht = self.X[5]
#         # print(_tht)
#         X_dot = [_x,
#                  _y,
#                  _tht,
#                  __x,
#                  __y,
#                  0
#                 ]
#         _X = np.multiply(delta_t, X_dot)
#         self.X += np.multiply(delta_t, X_dot)
#         self.X[5] = delta*self.lf*np.sqrt(self.X[3]**2+self.X[4]**2)

import os
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import numpy as np


class Car:
    # def __init__(self, x, y, angle=0.0, length=4, max_steering=30, max_acceleration=5.0):
    def __init__(self, X):
        self.position = Vector2(X[0], X[1])
        self.velocity = Vector2(0.0, 0.0)
        self.angle = X[2]
        self.length = 4
        self.max_acceleration = 5.0
        self.max_steering = np.pi/4
        self.max_velocity = 50
        self.brake_deceleration = 10
        self.free_deceleration = 2

        # self.acceleration = 0.0
        # self.steering = 0.0
        self.X = X

    def step(self, u, dt, skid=False):
        if skid:
            if u[0] != 0:
                u[0] = 1.5*u[0]/abs(u[0]) # skidding caps acceleration at 1.5
        print(u)
        self.velocity += (u[0] * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))
        if u[1]:
            turning_radius = self.length / sin(-u[1])
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += angular_velocity * dt
        self.X = np.array([self.position.x, self.position.y, self.angle, self.velocity.x, self.velocity.y, 0])
        # print(self.X)