import torch
import random
import numpy as np
from environment import Environment

class Actor(torch.nn.Module):
    def __init__(self, in_features=6+6, num_actions=5+5, init_weights=None): # X^-X, X*-X
        self.fc1 = torch.nn.Linear(in_features, 200)
        self.fc2 = torch.nn.Linear(200, num_actions)
        if init_weights is not None:
            self.fc1.weight.data = init_weights[0]
            self.fc2.weight.data = init_weights[1]
    
    def forward(self, X):
        return self.fc2(self.fc1(X))
    
    def backprop(self, X, Y):
        # do backprop
        pass

class Critic(torch.nn.Module):
    def __init__(self, in_state_features=6+6, in_action_features=5+5, init_weights=None): # X^-X, X*-X
        self.fc1 = torch.nn.Linear(in_state_features, 32)
        self.fc2 = torch.nn.Linear(32, 16)

        # Action as input
        self.fc3 = torch.nn.Linear(in_action_features, 32)

        self.fc4 = torch.nn.Linear(32, 200)
        self.fc5 = torch.nn.Linear(200, 200)
        self.out = torch.nn.Linear(200, 1)


        if init_weights is not None:
            self.fc1.weight.data = init_weights[0]
            self.fc2.weight.data = init_weights[1]
            self.fc3.weight.data = init_weights[2]
            self.fc4.weight.data = init_weights[3]
            self.fc5.weight.data = init_weights[4]
            self.out.weight.data = init_weights[5]
    
    def forward(self, X, u):
        mid = torch.cat((self.fc1(self.fc2(X)), self.fc3(u)))
        return self.out(self.fc5(self.fc4(mid)))
    
    def backprop(self, L):
        # do backprop
        loss.backward()
        optimizer.step()    # Does the update

# complete DDPG

class Buffer:
    def __init__(self, batch_size=50):
        self.capacity = 100000
        self.counter = 0
        self.states = np.zeros((self.capacity, 12))
        self.actions = np.zeros((self.capacity, 2))
        self.rewards = np.zeros((self.capacity, 1))
        self.next_states = np.zeros((self.capacity, 12))
    
    def record(self, observation):
        idx = self.counter % self.capacity
        self.states[idx] = observation[0]
        self.actions[idx] = observation[1]
        self.rewards[idx] = observation[2]
        self.next_states[idx] = observation[3]
        self.counter +=1

# train the model
_u = []
# 1.) Randomly initialize the critic and actor networks.
actor = Actor()
critic = Critic()
# 2.) Initialize the target models
target_actor = Actor()
target_critic = Critic()
# 3.) Initialize replay buffer
buffer = Buffer()
env = Environment()
N = 50
for episode in range(100000):
    # Resets initial state
    env.reset()
    for t in range(1000):
        u = actor.forward(env.car.X)
        # discretize this.
        u += 0.1*(random.random()-0.5)*u
        # execute input, get reward
        x = env.car.X # state
        r = env.step(u)
        _x = env.car.X # next state
        buffer.record([x, u, r, _x])
        if buffer.counter<N:
            idxs = np.random.choice(buffer.counter, buffer.counter)
        else:
            idxs = np.random.choice(buffer.counter, N)
        L = 0
        for i in idxs:
            a = target_actor.forward(buffer.next_states[i])
            y_i = buffer.rewards[i]+0.99*target_critic.forward(buffer.next_states[i], a)
            L += (y_i - critic.forward(buffer.states[i], buffer.actions[i]))**2
        L /= len(idxs)
        # update critic
        critic.backprop(L)
        # update actor
        # update targets


# complete DAgger