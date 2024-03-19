import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import datetime

# Define the discount rate for future rewards
GAMMA = 0.99

# Define the parameter for soft target network updates
TAU = 1e-3

# Define the learning rates for the actor and critic networks
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

# Define the initializers for the neural networks
def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

# Define the actor network architecture
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1=300, hidden2=600):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.apply(weight_init)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

# Define the critic network architecture
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden1=300, hidden2=600):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.apply(weight_init)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Define the Ornstein-Uhlenbeck Noise class
class OUNoise:
    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.action_size = action_size
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state

# Define the Gaussian Noise class
class GNoise:
    def __init__(self, action_size, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.action_size = action_size

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_size)

# Define the DDPG agent
class DDPGAgent:
    def __init__(self, state_size, action_size, seed):
        self.actor = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR)

        self.critic = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_CRITIC)

        self.noise = OUNoise(action_size, seed)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch
