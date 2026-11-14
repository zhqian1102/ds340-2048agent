# Code resources:
# PyTorch DQN tutorial: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# The training loop (epsilon decay, learning from the replay buffer each step, updating the target network) follows the structure of this tutorial.

# AI usage statement:
# Claude suggested the log2 reward scaling and the value of separate training seeds. 


import csv
import numpy as np
import torch
from game2048 import Game2048, encode_board
from dqn import QNetwork, ReplayBuffer, select_action, optimize


# Hyperparameters
N_EPISODES = 2000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0005 # learning rate
BUFFER_SIZE = 50000
TRAIN_START = 1000 # begin training after this many transitions are stored in the buffer
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 1000 # number of episodes over which epsilon decays from EPS_START to EPS_END
TARGET_UPDATE = 5 # number of episodes between target network updates


def compute_reward(reward, info):
    """
    Compress the reward to a smaller range. 
    """
    return float(np.log2(reward + 1))


# Initialization
net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(net.state_dict())
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)
log_file = open("training_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["episode", "score", "max_tile", "n_moves", "epsilon", "avg_loss"])


# Training loop
for episode in range(N_EPISODES):
    epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * episode / EPS_DECAY_EPISODES)
    game = Game2048(seed=100000 + episode) # use a different seed to avoid using the same random numbers 
    done = False
    losses = []

    # play one episode
    while not done:
        state = encode_board(game.board) # encode the current board state
        action = select_action(net, game.board, game.legal_actions(), epsilon) # select an action using epsilon greedy policy
        board, reward, done, info = game.step(action) # take the action in the game
        next_legal = [a in game.legal_actions() for a in range(4)]
        buffer.push(state, action, compute_reward(reward, info), encode_board(board), done, next_legal)

        # If we have enough transitions in the buffer, perform a training step
        if len(buffer) >= TRAIN_START:
            losses.append(optimize(net, target_net, optimizer, buffer, BATCH_SIZE, GAMMA))

    # Update the target network every TARGET_UPDATE episodes
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(net.state_dict())

    avg_loss = sum(losses) / len(losses) if losses else 0.0

    # Log the results of this episode
    writer.writerow([episode, game.score, game.max_tile(), game.n_moves, round(epsilon, 3), round(avg_loss, 4)])
    log_file.flush()

    # Print progress every 10 episodes
    if episode % 10 == 0:
        print(f"Episode {episode}: score={game.score}, max_tile={game.max_tile()}, epsilon={epsilon:.2f}, loss={avg_loss:.4f}")


# Save the trained model
log_file.close()
torch.save(net.state_dict(), "dqn.pt")
print("Training complete. Model saved to dqn.pt. Training log saved to training_log.csv.")