"""
python run.py                    # full experiment: 300 readers x 20 sessions x 20 seeds
python run.py --seeds 3          # quick check
python run.py --sessions 60      # longer horizon, where personalization starts to pay off

Prints a results table and saves plots and a JSON summary to results/.
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from buddy.experiment import run, summarize
from buddy.strategies import ALL_STRATEGIES

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readers", type=int, default=300)
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    summaries = []
    for cls in ALL_STRATEGIES:
        start = time.time()
        summaries.append(summarize([run(cls, args.readers, args.sessions, seed) for seed in range(args.seeds)]))
        print(f"  ran {cls.name} in {time.time() - start:.1f}s")

    default = next(s for s in summaries if s["strategy"] == "Fixed default")
    print(f"\n{args.readers} simulated readers x {args.sessions} sessions x {args.seeds} seeds")
    print(f"best possible success rate (if every reader always got their best settings): {default['best_possible_success']:.1%}\n")
    print(f"{'strategy':<48}{'regret/reader':>15}{'less regret':>12}{'success first 5':>16}{'success last 5':>16}{'near-best last 5':>18}")
    for s in summaries:
        change = 1 - s["cumulative_regret"] / default["cumulative_regret"]
        print(f"{s['strategy']:<48}{s['cumulative_regret']:>9.2f} ±{s['cumulative_regret_sd']:.2f}{change:>+12.0%}"
              f"{s['first5_success']:>16.1%}{s['last5_success']:>16.1%}{s['last5_near_best']:>18.1%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    for s in summaries:
        ax1.plot(np.arange(1, args.sessions + 1), s["regret_by_session"], label=s["strategy"])
        smooth = np.convolve(s["first5_regret_by_reader"], np.ones(15) / 15, mode="valid")
        ax2.plot(np.arange(len(smooth)) + 8, smooth, label=s["strategy"])
    ax1.set(title="Learning each reader: regret per session", xlabel="session number for a reader", ylabel="regret (lower is better)")
    ax2.set(title="Cold start: regret in a new reader's first 5 sessions", xlabel="reader arrival order", ylabel="regret (15-reader moving average)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(RESULTS, f"regret_{args.sessions}sessions.png"), dpi=150)

    with open(os.path.join(RESULTS, f"summary_{args.sessions}sessions.json"), "w") as f:
        json.dump({"config": vars(args), "results": [
            {k: v for k, v in s.items() if not isinstance(v, np.ndarray)} for s in summaries]}, f, indent=2)
    print(f"\nsaved {os.path.relpath(RESULTS)}/regret_{args.sessions}sessions.png and summary_{args.sessions}sessions.json")


if __name__ == "__main__":
    main()
