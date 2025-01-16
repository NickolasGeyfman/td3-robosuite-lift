import os
import torch as T 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name='critic', checkpoint_dir='tmp/td3', learning_rate=10e-3):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, fc1_dims)  # Initialize the first fully connected layer
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)  # Initialize the second fully connected layer
        self.q1 = nn.Linear(fc2_dims, 1)  # Initialize the last fully connected layer, and output the Q-value

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.005)
        self.device = T.device('mps' if T.backends.mps.is_available() else 'cpu')

        print(f"Created Critic Network on device: {self.device}")

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q1 = self.q1(action_value)

        return q1
        
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, learning_rate=10e-3, n_actions=2, name='actor', checkpoint_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # Initialize the first fully connected layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # Initialize the second fully connected layer
        self.output = nn.Linear(self.fc2_dims, self.n_actions)  # Initialize the output layer

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('mps' if T.backends.mps.is_available() else 'cpu')

        print(f"Created Actor Network on device: {self.device}")

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x) 
        x = self.fc2(x)
        x = F.relu(x)

        x = T.tanh(self.output(x))

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))