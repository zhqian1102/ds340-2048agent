import pandas as pd
from pathlib import Path
from evaluate_dqn import evaluate_dqn

def evaluate_models(n_games = 200):
    model_dir = Path("models")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    summaries = []

    for model_path in sorted(model_dir.glob("dqn_*.pt" )):
        model_name = model_path.stem
        print(f"Evaluating {model_name}")
        results = evaluate_dqn(n_games=n_games, model_path=str(model_path), output=str(output_dir / f"{model_name}.csv"))
        df = pd.DataFrame(results)
        summaries.append({
            "model": model_name,
            "games": len(df),
            "mean_score": df["score"].mean(),
            "median_score": df["score"].median(),
            "std_score": df["score"].std(),
            "best_score": df["score"].max(),
            "mean_moves": df["n_moves"].mean(),
            "pct_reached_256": (df["max_tile"] >= 256).mean() * 100,
            "pct_reached_512": (df["max_tile"] >= 512).mean() * 100,
            "pct_reached_1024": (df["max_tile"] >= 1024).mean() * 100,
            "pct_reached_2048": (df["max_tile"] >= 2048).mean() * 100,
        })
    if not summaries:
        print("No models found in the models/")
        return
    summary = pd.DataFrame(summaries).sort_values("mean_score", ascending=False)
    summary.to_csv(output_dir / "model_comparison.csv", index=False)    
    print("Model Comparison:")
    print(summary.round(2).to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
