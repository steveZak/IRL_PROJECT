import torch
import torch.nn.functional as F
import random
import numpy as np
import pygame
import time
from pygame.locals import *
from environment import Environment

tau = 0.2
# pygame.init()

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

class Buffer:
    def __init__(self, batch_size=50):
        self.capacity = 1000000
        self.counter = 0
        self.states = np.zeros((self.capacity, 12))
        self.actions = np.zeros((self.capacity, 6))
        self.rewards = np.zeros((self.capacity, 1)) # rewards not required for dagger
        self.next_states = np.zeros((self.capacity, 12))
    
    def record(self, observation):
        idx = self.counter % self.capacity
        self.states[idx] = observation[0]
        self.actions[idx] = observation[1]
        self.rewards[idx] = observation[2]
        self.next_states[idx] = observation[3]
        self.counter +=1

real_env = Environment()
# control = [[-5.0, -2.5, 0, 2.5, 5.0], [-np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3]]
control = [[-5.0, 0, 5.0], [-np.pi/3, 0, np.pi/3]]
actor = Actor()
act_opt = torch.optim.Adam(actor.parameters())
# problem
# need to include a trajectory with various dynamic changes.
# one is controlled by the actor
# combined trajectory is propagated forward to get X_hat-X
# but this cn be done after the demonstration loop

# demonstration part (no adaptation, since there is no control network at this stage and all info is known)
data = []
num_demonstrations = 20
for ep in range(num_demonstrations):
    config_g, config_x, config_icepatch, config_blowout = real_env.reset() # noise=False
    # getting expert trajectory
    # switch to while until whitespace
    traj_len = 0
    buffer = Buffer()
    while True:
        keys = pygame.key.get_pressed()
        # gas = [keys[K_1], keys[K_2], keys[K_3], keys[K_4], keys[K_5]]
        # steering = [keys[K_6], keys[K_7], keys[K_8], keys[K_9], keys[K_0]]
        gas = [keys[K_DOWN], 0.5, keys[K_UP]]
        steering = [keys[K_LEFT], 0.5, keys[K_RIGHT]]
        stop = keys[K_SPACE] == 1
        pygame.event.pump()
        X = np.array(real_env.car.X)
        if sum(steering) == 0:
            steer_sig = 0
        else:
            steer_sig = control[1][np.argmax(steering)]
        if sum(gas) == 0:
            gas_sig = 0
        else:
            gas_sig = control[0][np.argmax(gas)]
        u = [gas_sig, steer_sig]
        r = real_env.step(u)
        _X = np.array(real_env.car.X)
        if max(gas) == 0.5:
            gas = [0, 1, 0]
        else:
            _gas = [0, 0, 0]
            idx = np.argmax(gas)
            _gas[idx] = 1
            gas = _gas
        if max(steering) == 0.5:
            steering = [0, 1, 0]
        else:
            _steering = [0, 0, 0]
            idx = np.argmax(steering)
            _steering[idx] = 1
            steering = _steering
        gas.extend(steering)
        print(gas)
        buffer.record([np.concatenate(((X - X), (real_env.goal - X)), axis=0), gas, r, np.concatenate(((_X - _X), (real_env.goal - _X)), axis=0)])
        # buffer.record([, u, r, _X]) # rewards kinda irrelevant here, including bc why not
        traj_len += 1
        if stop:
            break
    time.sleep(1)
    data.append(buffer)
    # real_env.reset(X=config_x, goal=config_g, icepatch=config_p) # reset env to the previous config
pygame.quit()
# pre-training
# train on initial dataset

loss = torch.nn.BCEWithLogitsLoss()
# loss = torch.nn.MultiLabelSoftMarginLoss()
# loss = torch.nn.CrossEntropyLoss()
for epoch in range(15):
    l=0
    ctr=0
    for i in range(num_demonstrations):
        buffer = data[i]
        idxs = list(range(buffer.counter))
        random.shuffle(idxs)
        for j in idxs:
            act_opt.zero_grad()
            # loss.zero_grad()
            action_pred = actor.forward(torch.Tensor(buffer.states[j]))
            l += actor.backprop(loss(action_pred, torch.Tensor(buffer.actions[j])))
            ctr += 1
    print(l/ctr)
# training on new demonstration and combine the inputs
# (corrective actions are similar to tamer? - corrective actions (e.g. further left) or another demonstration with true labels based on the current state?)
# real_env = Environment()
# goal, state, icepatch = real_env.reset()
# control_env = Environment()
# for ep in range(num_demonstrations):
#     control_env.reset(noise=False, X=real_env.car.X, goal=goal, icepatch=None)
#     prev_traj = None
#     X = np.array(real_env.car.X)
#     for t in range(traj_len):
#         if prev_traj is None:
#             X_hat = np.array(real_env.car.X)
#         else:
#             X_hat = prev_traj[5][0] # keep the 5 step ahead projection for a more adequate comparison to ddpg
#         if t%5==0:
#             # predict X_hat trajectory with the actor on the control environment
#             traj = []
#             control_env.reset(noise=False, X=real_env.car.X, goal=real_env.goal, icepatch=None)
#             for step in range(160): # changed to 40, so the animation is less jagged and the expert input sequence more lengthy
#                 if step<154 and random.random() < 0.1*(10-ep):
#                     # generate the expert input
#                     keys = pygame.key.get_pressed()
#                     gas = [keys[K_1], keys[K_2], keys[K_3], keys[K_4], keys[K_5]]
#                     steering = [keys[K_6], keys[K_7], keys[K_8], keys[K_9], keys[K_0]]
#                     stop = keys[K_SPACE] == 1
#                     pygame.event.pump()
#                     if stop:
#                         break
#                     X = np.array(control_env.car.X)
#                     if sum(steering) == 0:
#                         steer_sig = 0
#                     else:
#                         steer_sig = control[1][np.argmax(steering)]
#                     if sum(gas) == 0:
#                         gas_sig = 0
#                     else:
#                         gas_sig = control[0][np.argmax(gas)]
#                     u_exp = [gas_sig, steer_sig]
#                     _X = np.array(real_env.car.X)
#                     # how is the next state below useful?
#                     gas.extend(steering) # this is u_exp
#                     u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
#                     r = control_env.step(u_act)
#                     buffer.record([np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0), gas, 0, np.concatenate(((X_hat - _X), (real_env.goal - _X)), axis=0)])
#                 # apply to the control env
#                 traj.append([X, u])
#                 loss = control_env.step(u)
#                 X = control_env.car.X
#             u = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
#             # apply to the control env
#             traj.append([X, u])
#             loss = control_env.step(u)
#             X = control_env.car.X
#             # prev_traj = traj
#         u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0)))
#         # u_act = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0)))
#         # apply to the real env
#         loss = real_env.step(u)

control_env = Environment(gui=False)
for ep in range(num_demonstrations):
    real_env = Environment()
    goal, state, icepatch, blowout = real_env.reset()
    control_env.reset(noise=False, X=real_env.car.X, goal=goal, icepatch=None, blowout=None)
    prev_traj = None
    X = np.array(real_env.car.X)
    t=0
    while True:
        if prev_traj is None:
            X_hat = np.array(real_env.car.X)
        else:
            X_hat = prev_traj[5][0] # keep the 5 step ahead projection for a more adequate comparison to ddpg
        if t%5==0:
            # predict X_hat trajectory with the actor on the control environment
            traj = []
            control_env.reset(noise=False, X=real_env.car.X, goal=real_env.goal, icepatch=None, blowout=None)
            # gets corrective feedback only once per timestep now
            keys = pygame.key.get_pressed()
            # gas = [keys[K_1], keys[K_2], keys[K_3], keys[K_4], keys[K_5]]
            # steering = [keys[K_6], keys[K_7], keys[K_8], keys[K_9], keys[K_0]]
            gas = [keys[K_DOWN], 0.5, keys[K_UP]]
            steering = [keys[K_LEFT], 0.5, keys[K_RIGHT]]
            stop = keys[K_SPACE] == 1
            pygame.event.pump()
            if stop:
                break
            X = np.array(control_env.car.X)
            if sum(steering) == 0:
                steer_sig = 0
            else:
                steer_sig = control[1][np.argmax(steering)]
            if sum(gas) == 0:
                gas_sig = 0
            else:
                gas_sig = control[0][np.argmax(gas)]
            u_exp = [gas_sig, steer_sig]
            _X = np.array(real_env.car.X)
            # how is the next state below useful?
            gas.extend(steering) # this is u_exp
            for step in range(6):
                # generate the expert input
                u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
                r = control_env.step(u_act, gui=False)
                buffer.record([np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0), gas, 0, np.concatenate(((X_hat - _X), (real_env.goal - _X)), axis=0)])
                # apply to the control env
                traj.append([X, u])
                X = control_env.car.X
            u = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
            # apply to the control env
            traj.append([X, u])
            loss = control_env.step(u, gui=False)
            X = control_env.car.X
            prev_traj = traj
        u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0)))
        # u_act = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0)))
        # apply to the real env
        loss = real_env.step(u_act)
        t+=1

loss = torch.nn.BCEWithLogitsLoss()
# loss = torch.nn.MultiLabelSoftMarginLoss()
# loss = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    l=0
    ctr=0
    for i in range(num_demonstrations):
        buffer = data[i]
        idxs = list(range(buffer.counter))
        random.shuffle(idxs)
        for j in idxs:
            act_opt.zero_grad()
            # loss.zero_grad()
            action_pred = actor.forward(torch.Tensor(buffer.states[j]))
            l += actor.backprop(loss(action_pred, torch.Tensor(buffer.actions[j])))
            ctr += 1
    print(l/ctr)


torch.save(actor.state_dict(), "actor_dagger.pt")

# control_env = Environment(gui=False)
# for ep in range(10):
#     real_env = Environment()
#     goal, state, icepatch, blowout = real_env.reset()
#     control_env.reset(noise=False, X=real_env.car.X, goal=goal, icepatch=None, blowout=None)
#     prev_traj = None
#     X = np.array(real_env.car.X)
#     t=0
#     while True:
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
#                 u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
#                 r = control_env.step(u_act, gui=False)
#                 # apply to the control env
#                 traj.append([X, u])
#                 X = control_env.car.X
#             u = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
#             # apply to the control env
#             traj.append([X, u])
#             loss = control_env.step(u, gui=False)
#             X = control_env.car.X
#             prev_traj = traj
#         u_act = actor.forward(torch.Tensor(np.concatenate(((X_hat - X), (real_env.goal - X)), axis=0)))
#         # u_act = actor.forward(torch.Tensor(np.concatenate(((X - X), (real_env.goal - X)), axis=0)))
#         # apply to the real env
#         loss = real_env.step(u_act)
#         t+=1