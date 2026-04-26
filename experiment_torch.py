import torch

from deephedging.run_train_torch import run_experiment


def run_many_experiments(n_runs=5):

    results = []

    for i in range(n_runs):

        print(f"run {i}")

        r = run_experiment()

        results.append(r["pnl"][-1])

    return results