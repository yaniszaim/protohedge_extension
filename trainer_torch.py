import numpy as np
import psutil
import torch
import torch.optim as optim
from collections import OrderedDict


def _weighted_mean(weights, x):
    return float(np.sum(weights * x))


def _weighted_err(weights, x):
    mean = _weighted_mean(weights, x)
    var = float(np.sum(weights * ((x - mean) ** 2)))
    return float(np.sqrt(var) / np.sqrt(len(weights)))


class TrainerTorch:
    def __init__(
        self,
        gym,
        lr=1e-3,
        device="cpu",
        clipvalue=None,
        global_clipnorm=None,
        lr_decay_factor=None,
        lr_decay_patience=None,
        lr_min=None,
        scheduler_monitor="val",
    ):
        self.gym = gym
        self.device = device
        self.optimizer = optim.Adam(self.gym.parameters(), lr=lr)
        self.best_state_dict = None
        self.clipvalue = None if clipvalue is None else float(clipvalue)
        self.global_clipnorm = None if global_clipnorm is None else float(global_clipnorm)
        self.scheduler_monitor = str(scheduler_monitor)
        self.scheduler = None
        if (
            lr_decay_factor is not None
            and lr_decay_patience is not None
            and float(lr_decay_factor) < 1.0
        ):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(lr_decay_factor),
                patience=int(lr_decay_patience),
                min_lr=0.0 if lr_min is None else float(lr_min),
            )

    @staticmethod
    def _clone_state_dict(module):
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in module.state_dict().items())

    def _slice_batch(self, obj, idx):
        if isinstance(obj, torch.Tensor):
            return obj[idx]
        if isinstance(obj, dict):
            return {k: self._slice_batch(v, idx) for k, v in obj.items()}
        if hasattr(obj, "keys"):
            return obj.__class__({k: self._slice_batch(v, idx) for k, v in obj.items()})
        return obj

    def train(
        self,
        train_data,
        val_data,
        train_weights,
        val_weights,
        n_epochs=200,
        epoch_refresh=20,
        batch_size=None,
    ):
        history = {
            "losses": {"batch": [], "training": [], "val": []},
            "losses_err": {"training": [], "val": []},
            "utilities": {
                "training_util": [],
                "training_util0": [],
                "training_util_err": [],
                "training_util0_err": [],
                "val_util": [],
                "val_util0": [],
            },
            "process": {"memory_rss": [], "memory_vms": []},
            "learning_rate": [],
            "best_epoch": -1,
            "best_loss": None,
            "init_loss": None,
            "init_loss_err": None,
        }

        process = psutil.Process()
        with process.oneshot():
            history["process"]["memory_rss"].append(process.memory_info().rss / (1024.0 * 1024.0))
            history["process"]["memory_vms"].append(process.memory_info().vms / (1024.0 * 1024.0))

        with torch.no_grad():
            init_result = self.gym.forward(train_data, training=False, return_paths=False)
        init_loss_path = init_result["loss_path"].detach().cpu().numpy()
        history["init_loss"] = _weighted_mean(train_weights, init_loss_path)
        history["init_loss_err"] = _weighted_err(train_weights, init_loss_path)
        history["best_loss"] = history["init_loss"]
        self.best_state_dict = self._clone_state_dict(self.gym)

        n_train = len(train_weights)
        effective_batch_size = 32 if batch_size is None else int(batch_size)

        for epoch in range(n_epochs):
            self.gym.train()
            permutation = torch.randperm(n_train)
            batch_losses = []
            batch_sizes = []

            for start in range(0, n_train, effective_batch_size):
                idx = permutation[start:start + effective_batch_size]
                batch = self._slice_batch(train_data, idx)
                self.optimizer.zero_grad()
                batch_result = self.gym.forward(batch, training=True, return_paths=False)
                batch_result["loss"].backward()
                if self.clipvalue is not None:
                    torch.nn.utils.clip_grad_value_(self.gym.parameters(), self.clipvalue)
                if self.global_clipnorm is not None:
                    torch.nn.utils.clip_grad_norm_(self.gym.parameters(), self.global_clipnorm)
                self.optimizer.step()
                batch_losses.append(float(batch_result["loss"].detach().cpu().item()))
                batch_sizes.append(int(idx.numel()))

            batch_loss = float(np.average(batch_losses, weights=batch_sizes))

            self.gym.eval()
            with torch.no_grad():
                train_result = self.gym.forward(train_data, training=False, return_paths=False)
                val_result = self.gym.forward(val_data, training=False, return_paths=False)

            train_loss_path = train_result["loss_path"].detach().cpu().numpy()
            val_loss_path = val_result["loss_path"].detach().cpu().numpy()
            train_utility = train_result["utility"].detach().cpu().numpy()
            train_utility0 = train_result["utility0"].detach().cpu().numpy()
            val_utility = val_result["utility"].detach().cpu().numpy()
            val_utility0 = val_result["utility0"].detach().cpu().numpy()

            train_loss = _weighted_mean(train_weights, train_loss_path)
            val_loss = _weighted_mean(val_weights, val_loss_path)
            history["losses"]["batch"].append(batch_loss)
            history["losses"]["training"].append(train_loss)
            history["losses"]["val"].append(val_loss)
            history["losses_err"]["training"].append(_weighted_err(train_weights, train_loss_path))
            history["losses_err"]["val"].append(_weighted_err(val_weights, val_loss_path))

            history["utilities"]["training_util"].append(_weighted_mean(train_weights, train_utility))
            history["utilities"]["training_util0"].append(_weighted_mean(train_weights, train_utility0))
            history["utilities"]["training_util_err"].append(_weighted_err(train_weights, train_utility))
            history["utilities"]["training_util0_err"].append(_weighted_err(train_weights, train_utility0))
            history["utilities"]["val_util"].append(_weighted_mean(val_weights, val_utility))
            history["utilities"]["val_util0"].append(_weighted_mean(val_weights, val_utility0))
            history["learning_rate"].append(float(self.optimizer.param_groups[0]["lr"]))

            if train_loss < history["best_loss"]:
                history["best_loss"] = train_loss
                history["best_epoch"] = epoch
                self.best_state_dict = self._clone_state_dict(self.gym)

            if self.scheduler is not None:
                monitor_value = val_loss if self.scheduler_monitor == "val" else train_loss
                self.scheduler.step(monitor_value)

            with process.oneshot():
                history["process"]["memory_rss"].append(process.memory_info().rss / (1024.0 * 1024.0))
                history["process"]["memory_vms"].append(process.memory_info().vms / (1024.0 * 1024.0))

            if epoch % epoch_refresh == 0:
                print(
                    f"epoch {epoch} | batch {batch_loss:.4f} | train {train_loss:.4f} "
                    f"| val {val_loss:.4f}"
                )

        if self.best_state_dict is not None:
            self.gym.load_state_dict(self.best_state_dict)

        return history
