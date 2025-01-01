import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class ReplayBuffer():
    # input shape is a tuple and unpack it and send it to np.zeros
    # we are on continuous dimension, so n_action is actually is the "number of components of an action"
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0  # where to put new mem
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.state_memory = np.zeros(self.mem_size, *input_shape)
        self.next_state_memory = np.zeros(self.mem_size, *input_shape)
        self.reward_memory = np.zeros(self.mem_size, n_actions)
        self.action_memory = np.zeros(self.mem_size, n_actions)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)  # done flags of environment

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.mem_size  # circular buffer
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_counter += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)  # ensure not to sample zeros
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        (
            T.FloatTensor(states).to(self.device),
            T.FloatTensor(actions).to(self.device),
            T.FloatTensor(next_states).to(self.device),
            T.FloatTensor(rewards).to(self.device),
            T.tensor(terminals).to(self.device)
        )

        return states, actions, rewards, next_states, terminals


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        print('save checkpoint')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('load checkpoint')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,n_actions, name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob) #buna gerek yok mu?

        pi = T.tanh(self.pi(prob)) # bounds -1 +1

        return pi

    def save_checkpoint(self):
        print('save checkpoint')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('load checkpoint')
        self.load_state_dict(T.load(self.checkpoint_file))