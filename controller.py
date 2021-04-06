import torch
import random
from car import Car
import matplotlib.pyplot as plt

# class Controller:
e = env()
T = []
# while True:
for i in range(16000):
    c.step([0.01, random.random()*0.2-0.1]) # [wheel angular speed, wheel yaw], delta_t
    print(c.X)
    T.append([c.X[0], c.X[1]])
# print(len(T))
for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], 'ro')
plt.show()