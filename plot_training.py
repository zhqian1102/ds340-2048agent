import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_training(log_path, output_dir, window):
    df = pd.read_csv(log_path)
    required = {'episode', 'score', 'max_tile', 'n_moves', 'epsilon', 'avg_loss'}
    output_dir.mkdir(parents=True, exist_ok=True)
    for column, ylabel, filename in [
        ('score', 'Final game score', 'training_score.png'),
        ('avg_loss', 'Average training loss', 'training_loss.png'),
        ('max_tile', 'Highest tile', 'training_max_tile.png'),
        ('epsilon', 'Epsilon', 'training_epsilon.png'),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df['episode'], df[column], alpha=.25, label='Per episode')
        if column != 'epsilon':
            smoothed = df[column].rolling(window, min_periods=1).mean()
            ax.plot(df['episode'], smoothed, label=f'{window}-episode moving average')
        ax.set(xlabel='Training episode', ylabel=ylabel, title=f'{ylabel} over training')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def compare(baseline_path, dqn_path, output_dir):
    baseline = pd.read_csv(baseline_path)
    dqn = pd.read_csv(dqn_path)
    df = pd.concat([baseline, dqn], ignore_index=True)
    seed_sets = [set(g['seed']) for _, g in df.groupby('agent')]
    if len(seed_sets) > 1 and not all(s == seed_sets[0] for s in seed_sets):
        raise ValueError('The baseline and DQN result CSVs must have the same set of seeds.')
    summary = df.groupby('agent').agg(
        games=('score', 'size'), mean_score=('score', 'mean'),
        median_score=('score', 'median'), mean_moves=('n_moves', 'mean'),
    )
    for tile in (256, 512, 1024, 2048):
        summary[f'pct_reached_{tile}'] = df.groupby('agent')['max_tile'].apply(
            lambda x: 100 * (x >= tile).mean())
    summary.to_csv(output_dir / 'agent_comparison.csv')
    print(summary.round(2).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    summary['mean_score'].plot.bar(ax=ax)
    ax.set(xlabel='Agent', ylabel='Mean final score', title='2048 agent performance')
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / 'agent_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='training_log.csv')
    parser.add_argument('--baselines', default='baseline_results.csv')
    parser.add_argument('--dqn', default='dqn_results.csv')
    parser.add_argument('--output-dir', default='plots')
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()
    out = Path(args.output_dir)
    plot_training(args.log, out, args.window)
    compare(args.baselines, args.dqn, out)
