import numpy as np
import matplotlib.pyplot as plt


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_training(results):
    """
    Plot training diagnostics for the current torch experiment result format,
    while still tolerating the older minimal dictionaries.
    """

    history = results.get("history")

    if history is None:
        loss = _as_numpy(results["loss"])
        pnl = _as_numpy(results["pnl"])

        plt.figure(figsize=(6, 4))
        plt.plot(loss)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(pnl)
        plt.title("PnL")
        plt.xlabel("Sample")
        plt.ylabel("PnL")
        plt.tight_layout()
        plt.show()
        return

    epochs = np.arange(1, len(history["losses"]["training"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs, history["losses"]["batch"], label="batch", alpha=0.6)
    ax.plot(epochs, history["losses"]["training"], label="train", linewidth=2)
    ax.plot(epochs, history["losses"]["val"], label="val", linewidth=2)
    if history.get("best_epoch", -1) >= 0:
        best_epoch = int(history["best_epoch"]) + 1
        best_loss = float(history["best_loss"])
        ax.scatter([best_epoch], [best_loss], c="black", marker="*", s=60, label="best")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    utils = history.get("utilities", {})
    if utils:
        ax.plot(epochs, utils.get("training_util", []), label="hedged, train", linewidth=2)
        ax.plot(epochs, utils.get("training_util0", []), label="unhedged, train", linestyle="--")
        ax.plot(epochs, utils.get("val_util", []), label="hedged, val", linewidth=2)
        ax.plot(epochs, utils.get("val_util0", []), label="unhedged, val", linestyle="--")
        ax.set_title("Utility")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Expected Utility")
        ax.legend()
    else:
        pnl = _as_numpy(results.get("pnl", results["training_result"]["pnl"]))
        ax.hist(pnl.reshape(-1), bins=50)
        ax.set_title("PnL")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
