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
    def __init__(self, in_features=6+6, num_actions=5+5, init_weights=None): # X^-X, X*-X
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 200)
        self.fc2 = torch.nn.Linear(200, num_actions)
        self.out = torch.nn.Softmax()
        if init_weights is not None:
            self.fc1.weight.data = init_weights[0]
            self.fc2.weight.data = init_weights[1]

    def update(self, model):
        self.fc1.weight.data = (1-tau)*self.fc1.weight.data + tau*model.fc1.weight.data
        self.fc2.weight.data = (1-tau)*self.fc2.weight.data + tau*model.fc2.weight.data

    def forward(self, X):
        outs = self.fc2(self.fc1(X))
        return torch.cat((self.out(outs[torch.LongTensor([0,1,2,3,4])]), self.out(outs[torch.LongTensor([5,6,7,8,9])])), dim=0)
    
    def backprop(self, loss):
        # do backprop
        loss.backward()
        act_opt.step()
        return loss.item()

class Buffer:
    def __init__(self, batch_size=50):
        self.capacity = 100000
        self.counter = 0
        self.states = np.zeros((self.capacity, 12))
        self.actions = np.zeros((self.capacity, 10))
        self.rewards = np.zeros((self.capacity, 1)) # rewards not required for dagger
        self.next_states = np.zeros((self.capacity, 12))
    
    def record(self, observation):
        idx = self.counter % self.capacity
        self.states[idx] = observation[0]
        self.actions[idx] = observation[1]
        self.rewards[idx] = observation[2]
        self.next_states[idx] = observation[3]
        self.counter +=1

# combines expert and actor actions (include tau weight later?)
def getCombinedAction(u_exp_idx, u_act): # gather class probabilities, instead of actor inputs
    u_exp = np.array([0,0,0,0,0,0,0,0,0,0])
    u_exp[int(u_exp_idx[0])] = 1
    u_exp[int(u_exp_idx[1])] = 1 # 5+? (for now doesn't matter, since turning is kinda garbage)
    # u_act = np.array([0,0,0,0,0,0,0,0,0,0])
    # u_act[control[0].index(u_act_idx[0])] = 1
    # u_act[5+control[1].index(u_act_idx[1])] = 1
    gas = u_exp[0:5] + u_act[0:5]/sum(u_act[0:5])
    steer = u_exp[5:10] + u_act[5:10]/sum(u_act[5:10])
    u_comb = np.array([0,0,0,0,0,0,0,0,0,0])
    u_comb[np.argmax(gas)] = 1
    u_comb[5+np.argmax(steer)] = 1
    return u_comb

real_env = Environment()
control = [[-100, -50, 0, 50, 100], [-2e-2, -1e-2, 0, 1e-2, 2e-2]]
actor = Actor()
act_opt = torch.optim.Adam(actor.parameters())
# problem
# need to include a trajectory with various dynamic changes.
# one is controlled by the actor
# combined trajectory is propagated forward to get X_-X
# but this cn be done after the demonstration loop

# demonstration part (no adaptation, since there is no control network at this stage and all info is known)
data = []
num_demonstrations = 10
for ep in range(num_demonstrations):
    config_g, config_x, config_p = real_env.reset(noise=False)
    # getting expert trajectory
    # switch to while until whitespace
    traj_len = 0
    buffer = Buffer()
    while True:
        keys = pygame.key.get_pressed()
        gas = [keys[K_1], keys[K_2], keys[K_3], keys[K_4], keys[K_5]]
        steering = [keys[K_6], keys[K_7], keys[K_8], keys[K_9], keys[K_0]]
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
        _X = np.array(real_env.car.X) # what is the error in this case
        # u_onehot = [0,0,0,0,0,0,0,0,0,0]
        # u_onehot[control[0].index(u[0])]=1
        # u_onehot[5+control[1].index(u[1])]=1
        gas.extend(steering)
        buffer.record([np.concatenate(((X - X), (real_env.goal - X)), axis=0), gas, r, np.concatenate(((_X - _X), (real_env.goal - _X)), axis=0)])
        # buffer.record([, u, r, _X]) # rewards kinda irrelevant here, including bc why not
        traj_len += 1
        if stop:
            break
    time.sleep(1)
    data.append(buffer)
    real_env.reset(X=config_x, goal=config_g, puddle=config_p) # reset env to the previous config
print("cool")
pygame.quit()
# pre-training
# train on initial dataset

loss = torch.nn.BCEWithLogitsLoss()
# loss = torch.nn.MultiLabelSoftMarginLoss()
# loss = torch.nn.CrossEntropyLoss()
for epoch in range(50):
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
control_env = Environment()
for ep in range(num_demonstrations):
    control_env.reset(noise=False, X=real_env.car.X)
    buffer = data[ep]
    prev_traj = None
    X = np.array(real_env.car.X)
    for t in range(traj_len):
        if prev_traj is None:
            X_ = np.array(real_env.car.X)
        else:
            X_ = prev_traj[5][0] # keep the 5 step ahead projection for a more adequate comparison to ddpg
        if t%5==0:
            # predict X_ trajectory with the actor on the control environment
            traj = []
            for step in range(6):
                act_opt.zero_grad()
                u_act = actor.forward(torch.Tensor(np.concatenate(((X_ - X), (real_env.goal - X)), axis=0))) # fix the inputs here and further down
                # u_act = [control[0][torch.argmax(u_act[:5])], control[1][torch.argmax(u_act[5:10])]]
                u = getCombinedAction(buffer.actions[t], u_act.detach().numpy())
                # apply to the control env
                traj.append([X, u])
                loss = control_env.step(u)
                X = control_env.car.X
                # training here
                actor.backprop(loss) # ensure gradients are conserved (don't detach)
            prev_traj = traj
        u_act = actor.forward(torch.Tensor(np.concatenate(((X_ - X), (real_env.goal - X)), axis=0)))
        # u_act = [control[0][torch.argmax(u_act[:5])], control[1][torch.argmax(u_act[5:10])]]
        u = getCombinedAction(buffer.actions[t], u_act.detach().numpy())
        # apply to the real env
        loss = real_env.step(u)