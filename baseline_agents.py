import numpy as np
from game2048 import move_board

#Chooses a random legal move
class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def choose_action(self, game):
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal actions available.") 
        return int(self.rng.choice(actions))

#Chooses the best move based on a simple heuristic (score gain + empty cells + corner bonus)
class RuleBasedAgent:
    def choose_action(self, game):
        actions = game.legal_actions()
        if not actions:
            raise ValueError("No legal actions available.") 
        best_action, best_value = None, float('-inf')
        for action in actions:
            next_board, score_gain, _ = move_board(game.board, action)
            empty_cells = int(np.count_nonzero(next_board == 0))
            largest_tile = int(next_board.max())
            corners = (
                next_board[0, 0],
                next_board[0, -1], 
                next_board[-1, 0],
                next_board[-1, -1]
            )
            corner_bonus = int(largest_tile in corners)
            value = score_gain + 10 * empty_cells + 5 * corner_bonus
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
    