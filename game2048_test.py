# AI use statement:
# We used Claude Code to draft the tests in this file. We reviewed and ran all of the test code and checked the game rules against the original 2048 game.

import numpy as np
from game2048 import Game2048, merge_row_left, move_board, encode_board, UP, RIGHT, DOWN, LEFT


# ---------- single-row merge rules ----------
def test_merge_basic():
    assert merge_row_left([2, 2, 0, 0])[0] == [4, 0, 0, 0]
    assert merge_row_left([0, 0, 0, 2])[0] == [2, 0, 0, 0]
    assert merge_row_left([4, 0, 0, 4])[0] == [8, 0, 0, 0]


def test_merge_only_once_per_move():
    # [2,2,4] -> [4,4], NOT [8]
    assert merge_row_left([2, 2, 4, 0])[0] == [4, 4, 0, 0]
    assert merge_row_left([2, 2, 2, 2]) == ([4, 4, 0, 0], 8, 2)
    assert merge_row_left([4, 4, 8, 8]) == ([8, 16, 0, 0], 24, 2)


def test_merge_leftmost_pair_first():
    assert merge_row_left([2, 2, 2, 0])[0] == [4, 2, 0, 0]


def test_no_merge():
    assert merge_row_left([2, 4, 2, 4]) == ([2, 4, 2, 4], 0, 0)


# ---------- four directions ----------
B = np.array([
    [2, 0, 2, 0],
    [0, 0, 0, 0],
    [2, 0, 0, 0],
    [0, 0, 0, 4],
])


def test_left():
    new, gain, _ = move_board(B, LEFT)
    assert new[0].tolist() == [4, 0, 0, 0]
    assert new[3].tolist() == [4, 0, 0, 0]
    assert gain == 4


def test_right():
    new, _, _ = move_board(B, RIGHT)
    assert new[0].tolist() == [0, 0, 0, 4]
    assert new[2].tolist() == [0, 0, 0, 2]


def test_up():
    new, gain, _ = move_board(B, UP)
    assert new[:, 0].tolist() == [4, 0, 0, 0]
    assert new[:, 3].tolist() == [4, 0, 0, 0]
    assert gain == 4


def test_down():
    new, _, _ = move_board(B, DOWN)
    assert new[:, 0].tolist() == [0, 0, 0, 4]
    assert new[:, 2].tolist() == [0, 0, 0, 2]


# ---------- environment behaviour ----------
def test_reset_has_two_tiles():
    g = Game2048(seed=1)
    assert (g.board > 0).sum() == 2
    assert set(g.board[g.board > 0]) <= {2, 4}


def test_step_spawns_one_tile():
    g = Game2048(seed=1)
    g.board = np.array([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    board, reward, done, info = g.step(LEFT)
    assert reward == 4 and info["moved"]
    assert (board > 0).sum() == 2  # merged tile + new tile
    assert board[0, 0] == 4


def test_illegal_move_does_nothing():
    g = Game2048(seed=1)
    g.board = np.array([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    before = g.board.copy()
    board, reward, done, info = g.step(LEFT)
    assert not info["moved"] and reward == 0
    assert np.array_equal(board, before)
    assert LEFT not in g.legal_actions() and UP not in g.legal_actions()


def test_game_over():
    g = Game2048(seed=1)
    g.board = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert g.is_done()
    g.board[0, 1] = 2  # now a merge exists
    assert not g.is_done()


def test_same_seed_same_game():
    def play(seed):
        g = Game2048(seed=seed)
        rng = np.random.default_rng(123)
        while not g.is_done():
            g.step(rng.choice(g.legal_actions()))
        return g.score, g.board.copy()

    s1, b1 = play(7)
    s2, b2 = play(7)
    assert s1 == s2 and np.array_equal(b1, b2)


def test_random_game_consistency():
    g = Game2048(seed=3)
    rng = np.random.default_rng(0)
    total = 0
    done = False
    while not done:
        _, r, done, info = g.step(rng.choice(g.legal_actions()))
        total += r
    assert total == g.score
    assert g.board.sum() > 0 and (g.board == 0).sum() == 0  # full board at game over


def test_encode_board():
    board = np.array([[0, 2, 4, 2048], [0] * 4, [0] * 4, [0] * 4])
    x = encode_board(board)
    assert x.shape == (16, 4, 4)
    assert x.sum() == 16  # exactly one channel hot per cell
    assert x[0, 0, 0] == 1 and x[1, 0, 1] == 1 and x[2, 0, 2] == 1 and x[11, 0, 3] == 1
