import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

# SAC Implementation (simplified version)
class SACNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Actor network outputs mean and log_std for each action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std for each action
        )

        # Two Q-networks for double Q-learning
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Q-value for each action
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Q-value for each action
        )

        # Target Q-networks
        self.target_q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.target_q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=100000):
        self.device = torch.device("cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim

        self.network = SACNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Separate optimizers for actor and critics
        self.actor_optimizer = optim.Adam(self.network.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.network.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.network.q2.parameters(), lr=lr)

        # Initialize target networks
        for target_param, param in zip(self.network.target_q1.parameters(), self.network.q1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.network.target_q2.parameters(), self.network.q2.parameters()):
            target_param.data.copy_(param.data)

        self.memory = deque(maxlen=buffer_size)

        # Temperature parameter
        self.target_entropy = -action_dim  # Target entropy is -|A|
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            output = self.network.actor(state)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            # Use reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()

            # Apply softmax and mask
            action_probs = F.softmax(action, dim=-1)
            mask = state[-self.action_dim:].bool()
            action_probs = action_probs * mask
            action_probs = action_probs / (action_probs.sum() + 1e-10)

            return action_probs.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=256):
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        action_batch = torch.LongTensor([t[1] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        done_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)

        # Update temperature parameter
        actor_output = self.network.actor(state_batch)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        sampled_actions = normal.rsample()
        log_probs = normal.log_prob(sampled_actions).sum(dim=-1)

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # Update critics
        with torch.no_grad():
            next_actor_output = self.network.actor(next_state_batch)
            next_mean, next_log_std = next_actor_output.chunk(2, dim=-1)
            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_actions = next_normal.rsample()
            next_log_probs = next_normal.log_prob(next_actions).sum(dim=-1)

            next_action_probs = F.softmax(next_actions, dim=-1)
            next_q1 = self.network.target_q1(next_state_batch)
            next_q2 = self.network.target_q2(next_state_batch)
            next_q = torch.min(next_q1, next_q2)
            next_q = (next_action_probs * next_q).sum(dim=1)
            target_q = reward_batch + (1 - done_batch) * self.gamma * (next_q - self.alpha * next_log_probs)

        # Get current Q estimates
        current_q1 = self.network.q1(state_batch)
        current_q2 = self.network.q2(state_batch)
        current_q1 = current_q1.gather(1, action_batch.unsqueeze(1)).squeeze()
        current_q2 = current_q2.gather(1, action_batch.unsqueeze(1)).squeeze()

        # Compute critic losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update actor
        actor_output = self.network.actor(state_batch)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        actions = normal.rsample()
        log_probs = normal.log_prob(actions).sum(dim=-1)

        action_probs = F.softmax(actions, dim=-1)
        q1 = self.network.q1(state_batch)
        q2 = self.network.q2(state_batch)
        q = torch.min(q1, q2)
        q = (action_probs * q).sum(dim=1)

        actor_loss = (self.alpha * log_probs - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.network.target_q1.parameters(), self.network.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.network.target_q2.parameters(), self.network.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item()
