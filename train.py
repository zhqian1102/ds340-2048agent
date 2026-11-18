# Code resources:
# PyTorch DQN tutorial: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# The training loop (epsilon decay, learning from the replay buffer each step, updating the target network) follows the structure of this tutorial.

# AI usage statement:
# Claude suggested the log2 reward scaling and the value of separate training seeds. 
# Used Claude Code to add code that enables reading configurations from command lines to enhance efficiency. 
# Consulted Claude Code to refine file saving logic. 


import csv
import numpy as np
import torch
from game2048 import Game2048, encode_board
from dqn import QNetwork, ReplayBuffer, select_action, optimize
import argparse # to read command line arguments (e.g. --reward R1)
import random
import os


# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--reward", type=str, default="R0")
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# Hyperparameters
N_EPISODES = args.episodes
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0005 # learning rate
BUFFER_SIZE = 50000
TRAIN_START = 1000 # begin training after this many transitions are stored in the buffer
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = N_EPISODES // 2 # number of episodes over which epsilon decays from EPS_START to EPS_END
TARGET_UPDATE = 5 # number of episodes between target network updates
RUN_NAME = f"{args.reward}_seed{args.seed}"


# Fix random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def compute_reward(reward, info, done, prev_max, version):
    """
    Change original score into reward. 
    
    Input:
    reward: this step's original score
    info: additional information returned by step() (e.g. number of empty cells, max tile...)
    done: whether game is done
    prev_max: the max tile before this step
    version: using which reward (R0, R1, R2, R3)
    """
    # all versions base R0
    r = float(np.log2(reward + 1))

    # R1: encourage saving empty cells
    if version == "R1":
        r += 0.1 * info["empty_cells"] # chose 0.1. There are maximum 16 empty cells, so max score added is 1.6. This is a reasonable number in comparison with regular received by merging cells. 
    # R2: discourage losing
    elif version == "R2":
        if done:
            r -= 10 # chose 10. Make losing signification enough. 
    # R3: encourage making new max tile
    elif version == "R3":
        if info["max_tile"] > prev_max:
            r += float(np.log2(info["max_tile"]))
    
    return r


# Initialization
net = QNetwork()
target_net = QNetwork()
target_net.load_state_dict(net.state_dict())
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_file = open(f"logs/training_log_{RUN_NAME}.csv", "w", newline="")
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
        prev_max = game.max_tile()
        action = select_action(net, game.board, game.legal_actions(), epsilon) # select an action using epsilon greedy policy
        board, reward, done, info = game.step(action) # take the action in the game
        next_legal = [a in game.legal_actions() for a in range(4)]
        r = compute_reward(reward, info, done, prev_max, args.reward)
        buffer.push(state, action, r, encode_board(board), done, next_legal)

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
        print(f"[{RUN_NAME}] Episode {episode}: score={game.score}, max_tile={game.max_tile()}, epsilon={epsilon:.2f}, loss={avg_loss:.4f}")


# Save the trained model
log_file.close()
torch.save(net.state_dict(), f"models/dqn_{RUN_NAME}.pt")
print(f"Training complete. Model saved to dqn.pt. Saved models/dqn_{RUN_NAME}.pt and logs/training_log_{RUN_NAME}.csv")