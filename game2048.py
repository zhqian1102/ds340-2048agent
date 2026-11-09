# AI usage statement:
# We used Claude Code to help review and suggest better design choices.

# Code resources:
# Original 2048 game (Gabriele Cirulli): https://github.com/gabrielecirulli/2048. Reference for the game rules (merging, new tile is 2 with p=0.9 and 4 with p=0.1).
# NumPy documentation: https://numpy.org/doc/stable/
# PyTorch DQN tutorial: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html. Reference for the reset()/step() environment interface used by the DQN.

import numpy as np

# Action ids
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]


def merge_row_left(row):
    """Slide one row to the left and merge. Returns (new_row, score_gain, n_merges)."""
    tiles = [x for x in row if x != 0]
    out = []
    gain = 0
    merges = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gain += tiles[i] * 2
            merges += 1
            i += 2  # each tile can only merge once per move
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (len(row) - len(out))
    return out, gain, merges


def move_board(board, action):
    """Apply an action to a board without spawning a tile.
    Rotate so the action becomes left, merge every row, rotate back.
    np.rot90 rotates counter-clockwise: up->k=1, right->k=2, down->k=3, left->k=0."""
    k = {LEFT: 0, UP: 1, RIGHT: 2, DOWN: 3}[action]
    rotated = np.rot90(board, k)
    new = np.zeros_like(rotated)
    gain = 0
    merges = 0
    for r in range(rotated.shape[0]):
        new[r], g, m = merge_row_left(rotated[r])
        gain += g
        merges += m
    return np.rot90(new, -k), gain, merges


class Game2048:
    def __init__(self, size=4, seed=None):
        self.size = size
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self.n_moves = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy()

    def add_random_tile(self):
        empty = np.argwhere(self.board == 0)
        r, c = empty[self.rng.integers(len(empty))]
        self.board[r, c] = 2 if self.rng.random() < 0.9 else 4

    def legal_actions(self):
        return [a for a in range(4) if not np.array_equal(move_board(self.board, a)[0], self.board)]

    def is_done(self):
        return len(self.legal_actions()) == 0

    def max_tile(self):
        return int(self.board.max())

    def step(self, action):
        """Returns (next_board, reward, done, info).
        Default reward = score gained this move. An illegal move leaves the board
        unchanged, spawns nothing, and gives reward 0 (info['moved'] is False)."""
        new_board, gain, merges = move_board(self.board, action)
        moved = not np.array_equal(new_board, self.board)
        if moved:
            self.board = new_board
            self.score += gain
            self.n_moves += 1
            self.add_random_tile()
        done = self.is_done()
        info = {
            "moved": moved,
            "score_gain": gain,
            "merges": merges,
            "empty_cells": int((self.board == 0).sum()),
            "max_tile": self.max_tile(),
            "score": self.score,
            "n_moves": self.n_moves,
        }
        return self.board.copy(), gain, done, info

    def render(self):
        print(f"score={self.score}  moves={self.n_moves}")
        for row in self.board:
            print(" ".join(f"{v:5d}" if v else "    ." for v in row))
        print()


def encode_board(board, n_channels=16):
    """One-hot encode by log2 for the DQN: channel 0 = empty, channel k = tile 2**k.
    Output shape (n_channels, 4, 4), float32."""
    exps = np.zeros_like(board)
    nz = board > 0
    exps[nz] = np.log2(board[nz]).astype(np.int64)
    return (np.arange(n_channels)[:, None, None] == exps[None]).astype(np.float32)


if __name__ == "__main__":
    # Play manually in the terminal: w/a/s/d, q to quit
    keys = {"w": UP, "d": RIGHT, "s": DOWN, "a": LEFT}
    game = Game2048(seed=0)
    game.render()
    while not game.is_done():
        k = input("move (w/a/s/d, q): ").strip()
        if k == "q":
            break
        if k in keys:
            game.step(keys[k])
            game.render()
    print("game over, score", game.score, "max tile", game.max_tile())
