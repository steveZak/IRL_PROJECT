import math
import numpy as np

class Car:
    def __init__(self, x, y):
        # toyota corolla
        self.fr = 0.9 # friction coefficient?
        self.m = 1066
        self.Caf = 0.1
        self.Csf = 0.1
        self.Car = 0.1
        self.Iz = 1706
        self.lf = 1.054
        self.lr = 2.372-1.054
        self.beta_x = 0 # road slope angle
        self.beta_y = 0 # road tilt angle
        self.reff = 1*0.264 # estimated wheel radius
        self.X = [x, y, np.pi/4, 0.000001, 0, 0] # x, y, tht, x_dot, y_dot, tht_dot
    
    def step(self, u, delta_t=0.1, fr=0.9):
        # page 38 in https://core.ac.uk/download/pdf/81577667.pdf with air resistance set to 0
        # X_dot = Ax+Bu
        self.fr = fr
        # X_dot = np.add(
        #          [self.X[3],\
        #          self.X[4],\
        #          self.X[5],\
        #          -self.X[4]*self.X[5]-np.sign(self.X[3])*(0.5*0*0*0*(self.X[4]**2)+self.fr*self.m*9.81+self.m*9.81*np.sin(self.beta_x))/self.m,\
        #          -self.X[3]*self.X[5]+(2*self.Car*(self.lr*self.X[5]-self.X[4])/self.X[3]-9.81*np.sign(self.beta_y))/self.m,\
        #          self.lr*(2*self.Car*(self.lr*self.X[5]-self.X[4])/(self.X[3]*self.m))/self.Iz]\
        #       , [
        #          0,
        #          0,
        #          0,
        #          (self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0]))*np.cos(u[1])-2*self.Caf*(u[1]+(self.lf*self.X[5]-self.X[4]/self.X[3]))*np.sin(u[1])+self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0])))/self.m,
        #          (self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0]))*np.sin(u[1])-2*self.Caf*(u[1]+(self.lf*self.X[5]-self.X[4]/self.X[3]))*np.cos(u[1]))/self.m,
        #          self.lf*(self.Csf*((2*self.reff*u[0]-2*self.X[3])/(self.reff*u[0]))*np.sin(u[1])-2*self.Caf*(u[1]+(self.lf*self.X[5]-self.X[4]/self.X[3]))*np.cos(u[1]))/self.Iz,
        #         ])
        # X_dot = [self.X[3]+u[0]*self.reff*math.cos(u[1]),\
        #          self.X[4]+u[0]*self.reff*math.sin(u[1]),\
        #          self.X[5]+(self.X[4]+u[0]*self.reff*math.sin(u[1]))/(self.lr+self.lf),\
        #          0,\
        #          0,\
        #          0]
        tau = u[0]
        delta = u[1]
        if np.abs(tau/self.reff)<self.m*9.81*self.fr:
            __x = tau/(self.m*self.reff)*math.cos(delta)
            __y = tau/(self.m*self.reff)*math.sin(delta)
        else:
            __x = 9.81*self.fr*math.cos(delta) # *m?
            __y = 9.81*self.fr*math.sin(delta)
        (__x, __y) = np.multiply([[math.cos(self.X[2]), -math.sin(self.X[2])],[math.sin(self.X[2]), math.cos(self.X[2])]], [__x,__y])
        __x = __x[0]
        __y = __y[0]
        _x = self.X[3]
        _y = self.X[4]
        _tht = self.X[5]
        X_dot = [_x,
                 _y,
                 _tht,
                 __x,
                 __y,
                 delta*self.lf/(self.lf+self.lr)
                ]
        _X = np.multiply(delta_t, X_dot)
        self.X += np.multiply(delta_t, X_dot)