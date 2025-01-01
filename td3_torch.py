import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy


class ReplayBuffer:
    # input shape is a tuple and unpack it and send it to np.zeros
    # we are on continuous dimension, so n_action is actually is the "number of components of an action"
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0  # where to put new mem
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.current_size = 0

        self.state_memory = np.zeros(self.mem_size, *input_dims)
        self.next_state_memory = np.zeros(self.mem_size, *input_dims)
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

        index += 1
        self.mem_counter = index
        self.current_size = min(self.mem_size, self.current_size + 1)

    def sample(self, batch_size):
        batch = np.random.choice(self.current_size, batch_size)  # ensure not to sample zeros

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return (
            T.FloatTensor(states).to(self.device),
            T.FloatTensor(actions).to(self.device),
            T.FloatTensor(next_states).to(self.device),
            T.FloatTensor(rewards).to(self.device),
            T.tensor(terminals).to(self.device)
        )


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.name = None
        self.checkpoint_file = None

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def set_name(self, name, chkpt_dir='tmp/td3'):
        # needed after deep copying the components
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

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
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = None
        self.checkpoint_file = None

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def set_name(self, name, chkpt_dir='tmp/td3'):
        # needed after deep copying the components
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)  # buna gerek yok mu?

        pi = T.tanh(self.pi(x))  # bounds -1 +1

        return pi

    def save_checkpoint(self):
        print('save checkpoint')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('load checkpoint')
        self.load_state_dict(T.load(self.checkpoint_file))


class TD3:
    def __init__(self, alpha, beta, input_dims, n_actions, capital_t, batch_size, discount, env, d, tau, noise_std,
                 initial_pure_exploration_limit):
        self.buffer = ReplayBuffer(int(1e6), input_dims, n_actions)

        self.actor = ActorNetwork(alpha, input_dims, 256, 256, n_actions)
        self.critic1 = CriticNetwork(beta, input_dims, 256, 256, n_actions)
        self.critic2 = CriticNetwork(beta, input_dims, 256, 256, n_actions)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.set_name("target_actor")
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic1.set_name("target_critic1")
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.target_critic2.set_name("target_critic2")

        self.actor.set_name("actor")
        self.critic1.set_name("critic1")
        self.critic2.set_name("critic2")

        self.capital_t = capital_t
        self.batch_size = batch_size
        self.discount = discount
        self.env = env
        self.d = d
        self.tau = tau
        self.noise_std = noise_std
        self.initial_pure_exploration_limit = initial_pure_exploration_limit
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

    def run(self):

        # start the environment
        state = self.env.reset()

        for t in range(self.capital_t):

            # select action w noise
            action = None

            if t < self.initial_pure_exploration_limit:
                action = self.env.action_space.sample()
            else:
                state = T.FloatTensor(state).to(self.actor.device)
                action = self.actor.forward(state).to(self.actor.device) + \
                         T.FloatTensor(np.random.normal(scale=self.noise_std)).to(self.actor.device)

            # observe reward and next state
            state, reward, next_state, terminal = self.env.step(action)

            # store transition tuple
            self.buffer.store_transition(state, action, reward, next_state, terminal)

            # sample mini batch
            states, actions, next_states, rewards, terminals = self.buffer.sample(self.batch_size)

            # noisy action from target actor
            noisy_target_actions = self.target_actor.forward(next_states) + \
                                   T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
            scaled_noisy_target_actions = T.clamp(noisy_target_actions, self.min_action[0],
                                                  self.max_action[
                                                      0])  # ensure noisy actions are in the bounds of action space

            # calculate target value
            q1_target = self.target_critic1.forward(next_states, scaled_noisy_target_actions)
            q2_target = self.target_critic2.forward(next_states, scaled_noisy_target_actions)

            # q1 = q1.view(-1)
            # q2 = q2.view(-1)

            target = reward + self.discount * (T.min(q1_target, q2_target)) * terminals

            # update critics
            q1 = self.critic1.forward(states, actions)
            q2 = self.critic2.forward(states, actions)

            # reset grad for each step
            self.critic1.optimizer.zero_grad()
            self.critic2.optimizer.zero_grad()

            # optimize
            critic_loss = F.mse_loss(target, q1) + F.mse_loss(target, q2)
            critic_loss.backward()
            self.critic1.optimizer.step()
            self.critic2.optimizer.step()

            if t % self.d == 0:
                # update actor by DDPG
                self.actor.optimizer.zero_grad()
                actor_q1_loss = self.critic1.forward(states, self.actor.forward(states))
                actor_loss = -T.mean(actor_q1_loss)
                actor_loss.backward()
                self.actor.optimizer.step()

                # update target network
                for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
