import torch
import random
import numpy as np
import pygame
from pygame.locals import *
from environment import Environment


tau = 0.2
# pygame.init()
# clock = pygame.time.Clock()

# keydown handler
def keydown(event):
    global paddle1_vel, paddle2_vel, label

    if event.key == K_UP:
        paddle2_vel = -20
        label = 2
    elif event.key == K_DOWN:
        paddle2_vel = 20
        label = 1
    print(label)
    # elif event.key == K_w:
    #     paddle1_vel = -8
    # elif event.key == K_s:
    #     paddle1_vel = 8


# keyup handler
def keyup(event):
    global paddle1_vel, paddle2_vel, label

    if event.key in (K_w, K_s):
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        paddle2_vel = 0
        label = 0

class Actor(torch.nn.Module):
    def __init__(self, in_features=6+6, num_actions=5+5, init_weights=None): # X^-X, X*-X
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 200)
        self.fc2 = torch.nn.Linear(200, num_actions)
        if init_weights is not None:
            self.fc1.weight.data = init_weights[0]
            self.fc2.weight.data = init_weights[1]

    def update(self, model):
        self.fc1.weight.data = (1-tau)*self.fc1.weight.data + tau*model.fc1.weight.data
        self.fc2.weight.data = (1-tau)*self.fc2.weight.data + tau*model.fc2.weight.data

    def forward(self, X):
        return self.fc2(self.fc1(X))
    
    def backprop(self, loss):
        # do backprop
        loss.backward()
        act_opt.step()


class Buffer:
    def __init__(self, batch_size=50):
        self.capacity = 100000
        self.counter = 0
        self.states = np.zeros((self.capacity, 12))
        self.actions = np.zeros((self.capacity, 2))
        self.rewards = np.zeros((self.capacity, 1)) # rewards not required for dagger
        self.next_states = np.zeros((self.capacity, 12))
    
    def record(self, observation):
        idx = self.counter % self.capacity
        self.states[idx] = observation[0]
        self.actions[idx] = observation[1]
        self.rewards[idx] = observation[2]
        self.next_states[idx] = observation[3]
        self.counter +=1


env = Environment()
control = [[-100, -50, 0, 50, 100], [-2e-2, -1e-2, 0, 1e-2, 2e-2]]
actor = Actor()
act_opt = torch.optim.Adam(actor.parameters())
buffer = Buffer()
for epoch in range(10):
    config = env.reset()
    # getting expert trajectory
    # switch to while until whitespace
    for t in range(10000):
        keys = pygame.key.get_pressed()
        steering = [keys[K_1], keys[K_2], keys[K_3], keys[K_4], keys[K_5]]
        gas = [keys[K_6], keys[K_7], keys[K_8], keys[K_9], keys[K_0]]
        pygame.event.pump()
        X = env.car.X
        if sum(steering) == 0:
            steer_sig = 0
        else:
            steer_sig = control[0][np.argmax(steering)]
        if sum(gas) == 0:
            gas_sig = 0
        else:
            gas_sig = control[1][np.argmax(gas)]
        u = [steer_sig, gas_sig]
        r = env.step(u)
        _X = env.car.X # what is the error in this case
        buffer.record([np.concatenate(((X_ - X), (env.goal - X)), axis=0), u, r, np.concatenate(((X_ - X), (env.goal - _X)), axis=0)]) # rewards kinda irrelevant here, including bc why not
    env.reset(config) # reset env to the previous config
    # play out the combined trajectory
    prev_traj = []
    X_ = X.copy()
    if prev_traj is not None:
        X_ = prev_traj[5][0]
    else:
        X_ = X
    traj = []
    for t in range(10000):
        if t%5 == 0:
            # run this 6 timesteps ahead every 5 timesteps to get X_ (estimated X)
            # for i in range(6):
            u = actor.forward(torch.Tensor(np.concatenate(((X_ - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
            u = [control[0][torch.argmax(u[:5])], control[1][torch.argmax(u[5:10])]]
        # and propagate in the real env on every timestep

    # update actor based on the final loss of the combined actor-demonstrator trajectory.
    actor.backprop(loss)
    
    for t in range(10000):

    
        

# while True:
#     keys = pygame.key.get_pressed()
#     thrust = 0
#     yaw = 0
#     if keys[K_LEFT]:
#         yaw = 0.0001
#     if keys[K_RIGHT]:
#         yaw = -0.0001
#     if keys[K_UP]:
#         thrust = 40
#     if keys[K_DOWN]:
#         thrust = -40
#     if sum(keys) == 0:
#         print("ey")
#     else:
#         print("ey1")
#     pygame.event.pump()
