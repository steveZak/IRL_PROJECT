import torch
import torch.nn.functional as F
import random
import numpy as np
import pygame
import time
from pygame.locals import *
from environment import Environment

class Actor(torch.nn.Module):
    def __init__(self, in_features=6+6, num_actions=3+3, init_weights=None): # X^-X, X*-X
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 200)
        self.drop1 = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(200, num_actions)
        self.drop2 = torch.nn.Dropout()
        self.out = torch.nn.Softmax()
        if init_weights is not None:
            self.fc1.weight.data = init_weights[0]
            self.fc2.weight.data = init_weights[1]

    def update(self, model):
        self.fc1.weight.data = (1-tau)*self.fc1.weight.data + tau*model.fc1.weight.data
        self.fc2.weight.data = (1-tau)*self.fc2.weight.data + tau*model.fc2.weight.data

    def forward(self, X):
        outs = self.drop2(self.fc2(self.drop1(self.fc1(X))))
        return torch.cat((self.out(outs[torch.LongTensor([0,1,2])]), self.out(outs[torch.LongTensor([3,4,5])])), dim=0)
    
    def backprop(self, loss):
        # do backprop
        loss.backward()
        act_opt.step()
        return loss.item()

control = [[-5.0, 0, 5.0], [-np.pi/3, 0, np.pi/3]]
# actor=Actor()
# actor.load_state_dict(torch.load('actor_dagger_.pt')) # dagger sort of works
# actor.eval()
# min_l = -1e9
# min_losses = []
# control_env = Environment(gui=False)
# configs = []
# for ep in range(100):
#     print(ep)
#     real_env = Environment(gui=True)
#     goal, state, icepatch, blowout = real_env.reset()
#     configs.append([goal, state, icepatch, blowout])
#     control_env.reset(noise=False, X=real_env.car.X, goal=goal, icepatch=None, blowout=None)
#     prev_traj = None
#     X = np.array(real_env.car.X)
#     t=0
#     for i in range(1000):
#         if prev_traj is None:
#             X_hat = np.array(real_env.car.X)
#         else:
#             X_hat = prev_traj[5][0] # keep the 5 step ahead projection for a more adequate comparison to ddpg
#         if t%5==0:
#             # predict X_hat trajectory with the actor on the control environment
#             traj = []
#             control_env.reset(noise=False, X=real_env.car.X, goal=real_env.goal, icepatch=None, blowout=None)
#             _X = np.array(real_env.car.X)
#             # how is the next state below useful?
#             for step in range(6):
#                 # generate the expert input
#                 u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0))).detach().numpy()# fix the inputs here and further down
#                 r = control_env.step([control[0][np.argmax(u_act[0:3])], control[1][np.argmax(u_act[3:])]], gui=False)
#                 # apply to the control env
#                 traj.append([X, u_act])
#                 X = control_env.car.X
#             u = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0))).detach().numpy() # fix the inputs here and further down
#             # apply to the control env
#             traj.append([X, u])
#             loss = control_env.step([control[0][np.argmax(u[0:3])], control[1][np.argmax(u[3:])]], gui=False)
#             X = control_env.car.X
#             prev_traj = traj
#         u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0))).detach().numpy()
#         # u_act = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0)))
#         # apply to the real env
#         loss = real_env.step([control[0][np.argmax(u_act[0:3])], control[1][np.argmax(u_act[3:])]], gui=True)
#         if loss>min_l:
#             min_l = loss
#         # if loss>-9000:
#         #     break
#         t+=1
#     min_losses.append(min_l)
# print(np.mean(min_losses))

# # print(configs[:5])

actor=Actor()
actor.load_state_dict(torch.load('actor.pt')) # ddpg with early stopping
actor.eval()
min_l = -1e9
min_losses = []
control_env = Environment(gui=False)
for ep in range(100):
    # print(ep)
    real_env = Environment(gui=True)
    # goal, state, icepatch, blowout = real_env.reset(noise=True, X=configs[ep][1], goal=configs[ep][0], icepatch=configs[ep][2], blowout=configs[ep][3])
    goal, state, icepatch, blowout = real_env.reset(noise=True)
    # if ep<5:
    #     print(goal)
    #     print(state)
    control_env.reset(noise=False, X=real_env.car.X, goal=goal, icepatch=None, blowout=None)
    prev_traj = None
    X = np.array(real_env.car.X)
    t=0
    for i in range(1000):
        if prev_traj is None:
            X_hat = np.array(real_env.car.X)
        else:
            X_hat = prev_traj[5][0] # keep the 5 step ahead projection for a more adequate comparison to ddpg
        if t%5==0:
            # predict X_hat trajectory with the actor on the control environment
            traj = []
            control_env.reset(noise=False, X=real_env.car.X, goal=real_env.goal, icepatch=None, blowout=None)
            _X = np.array(real_env.car.X)
            # how is the next state below useful?
            for step in range(6):
                # generate the expert input
                u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0))).detach().numpy()# fix the inputs here and further down
                r = control_env.step([control[0][np.argmax(u_act[0:3])], control[1][np.argmax(u_act[3:])]], gui=False)
                # apply to the control env
                traj.append([X, u_act])
                X = control_env.car.X
            u = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0))).detach().numpy() # fix the inputs here and further down
            # apply to the control env
            traj.append([X, u])
            loss = control_env.step([control[0][np.argmax(u[0:3])], control[1][np.argmax(u[3:])]], gui=False)
            X = control_env.car.X
            prev_traj = traj
        u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0))).detach().numpy()
        # u_act = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0)))
        # apply to the real env
        loss = real_env.step([control[0][np.argmax(u_act[0:3])], control[1][np.argmax(u_act[3:])]], gui=True)
        if loss>min_l:
            min_l = loss
        # if loss>-9000:
        #     break
        t+=1
    min_losses.append(min_l)
print(np.mean(min_losses))
# -5096.870073471183
# -7338.781812035322

# -1302.1862403192283
# -1691.1404464721645
