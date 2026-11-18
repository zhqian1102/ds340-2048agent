import csv
import argparse
import numpy as np

from game2048 import Game2048
from evaluate_baselines import run_game
from dqn import DQNAgent

#evaluate dqn on same game seeds as baseline agents
def evaluate_dqn(n_games = 200, model_path = "dqn.pt", output = "dqn_results.csv"):
    dqn_agent = DQNAgent(model_path=model_path)
    results = []
    for seed in range(n_games):
        results.append({
            "agent": "DQNAgent",
            "seed": seed,
                **run_game(dqn_agent, seed)
            })
    with open(output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['agent', 'seed', 'score', 'max_tile', 'n_moves'])
        writer.writeheader()
        writer.writerows(results)
    scores = [r["score"] for r in results]
    max_tiles = np.array([r["max_tile"] for r in results])
    n_moves = [r["n_moves"] for r in results]
    print(f"  Average score: {np.mean(scores):.2f}")
    print(f"  Median score: {np.median(scores):.2f}")
    print(f"  Best score: {np.max(scores):.2f}")
    print(f"  Average number of moves: {np.mean(n_moves):.2f}")
    for tile in [256, 512, 1024, 2048]:
        percentage = np.mean(max_tiles >= tile) * 100
        print(f"Reached {tile}: {percentage:.1f}% of games")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--output", type=str, default="dqn_results.csv")
    parser.add_argument("--model", type=str, default="dqn.pt")

    args = parser.parse_args()
    evaluate_dqn(args.games, args.model, args.output)