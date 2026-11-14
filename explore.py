from game2048 import Game2048, encode_board
import torch
from dqn import QNetwork

game = Game2048(seed=0)
game.render()
x = encode_board(game.board)
print("shape:", x.shape)
print("layer 0:")
print(x[0])
print("layer 1:")
print(x[1])

net = QNetwork()
x_torch = torch.tensor(x).unsqueeze(0)
print("network input shape:", x_torch.shape)
q = net(x_torch)
print("Q-values on four directions:", q)