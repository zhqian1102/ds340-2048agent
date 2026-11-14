# Code resources:
# PyTorch DQN tutorial: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# ReplayBuffer, select_action (epsilon greedy), optimize (Bellman target, Huber loss, target network)
# Followed the structure of this tutorial, adapted for 2048 (one-hot board input, illegal-action masking).


import torch
import torch.nn as nn
import random
from collections import deque
from game2048 import encode_board
import numpy as np

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(16 * 4 * 4, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_legal):
        """
        Store a transition in the replay buffer. 
        """
        self.memory.append((state, action, reward, next_state, done, next_legal))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the replay buffer. 
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current size of the replay buffer. 
        """
        return len(self.memory)


def select_action(net, board, legal, epsilon):
    """
    Select an action using epsilon greedy policy.
    """
    if random.random() < epsilon:
        return random.choice(legal)

    x = torch.tensor(encode_board(board)).unsqueeze(0)

    # Only compute Q values, don't backpropagate, runs faster
    with torch.no_grad():
        q = net(x)[0] # get 4 numbers for 4 directions

    # Check legal actions and set Q values of illegal actions to -inf
    for a in range(4):
        if a not in legal:
            q[a] = float("-inf")

    return int(torch.argmax(q))


def optimize(net, target_net, optimizer, buffer, batch_size, gamma):
    """
    Sample a batch of transitions from the replay buffer and perform a single optimization step on the Q network. 
    """
    # Sample a batch of transitions from the replay buffer
    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones, next_legals = zip(*batch)
    states = torch.tensor(np.array(states))
    actions = torch.tensor(actions)  
    rewards = torch.tensor(rewards, dtype=torch.float32)   
    next_states = torch.tensor(np.array(next_states))    
    dones = torch.tensor(dones)       
    next_legals = torch.tensor(next_legals)  

    # Compute Q values for current states and actions
    q_all = net(states) # shape: (batch_size, 4)
    q_pred = q_all[torch.arange(batch_size), actions] # shape: (batch_size,)

    # Compute target Q values for next states using the target network
    with torch.no_grad():
        q_next = target_net(next_states)
        q_next[~next_legals] = float("-inf")
        best_next = q_next.max(dim=1).values
        best_next[dones] = 0.0
        target = rewards + gamma * best_next

    # Compute loss and perform backpropagation
    loss = nn.functional.smooth_l1_loss(q_pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


class DQNAgent:
    def __init__(self, model_path="dqn.pt"):
        self.net = QNetwork()
        self.net.load_state_dict(torch.load(model_path))

    def choose_action(self, game):
        return select_action(self.net, game.board, game.legal_actions(), 0.0)