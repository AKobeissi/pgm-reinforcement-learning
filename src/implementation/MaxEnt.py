import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class MaxEntNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Policy network for discrete actions
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.action_dim = action_dim

    def policy(self, state):
        """
        For a discrete action space:
         1) Compute logits from the policy network.
         2) Convert logits to probabilities via softmax.
         3) Compute log_probs and entropy for the distribution.
        """
        logits = self.policy_net(state)                 # [batch_size, action_dim]
        dist = F.softmax(logits, dim=-1)               # [batch_size, action_dim] in [0,1]
        log_probs = torch.log(dist + 1e-10)            # Avoid log(0)
        entropy = -(dist * log_probs).sum(dim=-1)      # Discrete entropy: -Σ p(a) log p(a)

        return dist, entropy

    def value(self, state):
        return self.value_net(state)

class MaxEntRL:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3, gamma=0.95, tau=0.1, alpha=0.1):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim

        # Force CPU usage
        self.device = torch.device("cpu")

        self.network = MaxEntNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = MaxEntNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)  # Smaller memory

    def select_action(self, state):
        """
        1) Get the discrete distribution from the current policy.
        2) apply a mask for invalid actions, if your environment needs it.
        3) Sample an action using torch.multinomial() or argmax() if you want a greedy action.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            # state shape => [1, state_dim]

            dist, _ = self.network.policy(state_t)  # dist shape => [1, action_dim]
            dist = dist.squeeze(0)                 # => [action_dim]

            # If you are masking invalid actions:
            # Here, assume the last `action_dim` elements of `state`
            # (i.e., state[-self.action_dim:]) are boolean flags
            # that indicate valid/invalid actions:
            mask = torch.FloatTensor(state[-self.action_dim:]).bool().to(self.device)
            dist = dist * mask
            dist = dist / (dist.sum() + 1e-10)  # Renormalize after masking

            # Sample an action
            action = torch.multinomial(dist, 1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        action_batch = torch.LongTensor([t[1] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_state_batch = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        done_batch = torch.FloatTensor([t[4] for t in batch]).to(self.device)

        # Compute target values
        with torch.no_grad():
            next_value = self.target_network.value(next_state_batch).squeeze(-1)  # shape: [batch_size]
            expected_value = reward_batch + self.gamma * next_value * (1 - done_batch)

        # Current value
        curr_value = self.network.value(state_batch).squeeze(-1)  # shape: [batch_size]
        value_loss = F.mse_loss(curr_value, expected_value)

        # Policy computation
        dist, entropy = self.network.policy(state_batch)  # shape: [batch_size, action_dim], [batch_size]
        log_prob = torch.log(dist + 1e-10)                # [batch_size, action_dim]

        # Advantage
        advantage = (expected_value - curr_value).detach()  # [batch_size]

        # Policy loss:
        #   We take log_prob for the actions actually taken, multiply by advantage, and average
        policy_loss = -(log_prob[range(batch_size), action_batch] * advantage).mean()

        # Entropy loss: negative of the mean entropy for a maximum entropy approach
        entropy_loss = -entropy.mean()

        # Combine losses
        total_loss = value_loss + policy_loss + self.alpha * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return total_loss.item()
