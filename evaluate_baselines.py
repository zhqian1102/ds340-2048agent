import csv
import argparse
import numpy as np

from game2048 import Game2048
from baseline_agents import RandomAgent, RuleBasedAgent

#play one game and return its results
def run_game(agent, seed):
    game = Game2048(seed=seed)
    while not game.is_done():
        action = agent.choose_action(game)
        board, reward, done, info = game.step(action)
        assert info["moved"], f"Agent chose illegal action {action} on board:\n{game.board}"
        if done:
            break

    return {
        "score": game.score,
        "max_tile": game.max_tile(),
        "n_moves": game.n_moves
    }

#evaluate both agents on the same game seeds
def evaluate_agents(n_games = 200, output = "baseline_results.csv"):
    results = []
    agents = [
    ("RandomAgent", RandomAgent),
    ("RuleBasedAgent", RuleBasedAgent)
    ]
    for agent_name, agent_class in agents:
        for seed in range(n_games):
            if agent_class is RandomAgent:
                agent = agent_class(seed=seed + 10000)
            else:
                agent = agent_class()
            game_result = run_game(agent, seed)
            results.append({
                "agent": agent_name,
                "seed": seed,
                **game_result
            })
    with open(output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['agent', 'seed', 'score', 'max_tile', 'n_moves'])
        writer.writeheader()
        writer.writerows(results)
    for agent_name, agent_class in agents:
        agent_results = [r for r in results if r["agent"] == agent_name]
        scores = [r["score"] for r in agent_results]
        max_tiles = np.array([r["max_tile"] for r in agent_results])
        n_moves = [r["n_moves"] for r in agent_results]
        print(f"{agent_name}:")
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
    parser.add_argument(
        "--games",
        type=int,
        default=200
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_results.csv"
    )
    args = parser.parse_args()
    evaluate_agents(
        n_games=args.games,
        output=args.output
    )