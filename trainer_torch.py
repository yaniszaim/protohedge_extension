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


def _result_diagnostics(result, market):
    actions = result["actions"].detach().cpu().numpy()
    deltas = result["deltas"].detach().cpu().numpy()
    diag = {
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "delta_abs_mean": float(np.mean(np.abs(deltas))),
        "pct_at_any_position_bound": 0.0,
        "pct_paths_touch_any_position_bound": 0.0,
    }
    if "ubnd_delta" in market and "lbnd_delta" in market:
        ubnd_delta = market["ubnd_delta"].detach().cpu().numpy()
        lbnd_delta = market["lbnd_delta"].detach().cpu().numpy()
        near_upper = np.isclose(deltas, ubnd_delta, atol=1e-4, rtol=0.0)
        near_lower = np.isclose(deltas, lbnd_delta, atol=1e-4, rtol=0.0)
        near_bound = near_upper | near_lower
        diag["pct_at_any_position_bound"] = float(np.mean(near_bound))
        diag["pct_paths_touch_any_position_bound"] = float(np.mean(np.any(near_bound, axis=(1, 2))))
    return diag


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
        selection_metric="train_loss",
        selection_alpha_action_abs=0.0,
        selection_alpha_delta_abs=0.0,
        selection_alpha_bound_occupancy=0.0,
        selection_alpha_path_bound_touch=0.0,
    ):
        selection_metric = str(selection_metric)

        def _selection_score(base_train_loss, base_val_loss, val_diag):
            if selection_metric == "train_loss":
                base = float(base_train_loss)
            elif selection_metric == "val_loss":
                base = float(base_val_loss)
            else:
                raise ValueError(f"Unsupported selection_metric '{selection_metric}'")
            return (
                base
                + float(selection_alpha_action_abs) * float(val_diag["action_abs_mean"])
                + float(selection_alpha_delta_abs) * float(val_diag["delta_abs_mean"])
                + float(selection_alpha_bound_occupancy) * float(val_diag["pct_at_any_position_bound"])
                + float(selection_alpha_path_bound_touch) * float(val_diag["pct_paths_touch_any_position_bound"])
            )

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
            "diagnostics": {
                "train_action_abs_mean": [],
                "val_action_abs_mean": [],
                "train_delta_abs_mean": [],
                "val_delta_abs_mean": [],
                "train_pct_at_any_position_bound": [],
                "val_pct_at_any_position_bound": [],
                "train_pct_paths_touch_any_position_bound": [],
                "val_pct_paths_touch_any_position_bound": [],
            },
            "learning_rate": [],
            "selection_metric": selection_metric,
            "selection_scores": {"val": []},
            "best_epoch": -1,
            "best_loss": None,
            "best_score": None,
            "init_loss": None,
            "init_loss_err": None,
        }

        process = psutil.Process()
        with process.oneshot():
            history["process"]["memory_rss"].append(process.memory_info().rss / (1024.0 * 1024.0))
            history["process"]["memory_vms"].append(process.memory_info().vms / (1024.0 * 1024.0))

        with torch.no_grad():
            init_result = self.gym.forward(train_data, training=False, return_paths=False)
            init_val_result = self.gym.forward(val_data, training=False, return_paths=False)
        init_loss_path = init_result["loss_path"].detach().cpu().numpy()
        init_val_loss_path = init_val_result["loss_path"].detach().cpu().numpy()
        history["init_loss"] = _weighted_mean(train_weights, init_loss_path)
        history["init_loss_err"] = _weighted_err(train_weights, init_loss_path)
        history["best_loss"] = history["init_loss"]
        init_val_loss = _weighted_mean(val_weights, init_val_loss_path)
        init_val_diag = _result_diagnostics(init_val_result, val_data["market"])
        history["best_score"] = _selection_score(history["init_loss"], init_val_loss, init_val_diag)
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
            train_diag = _result_diagnostics(train_result, train_data["market"])
            val_diag = _result_diagnostics(val_result, val_data["market"])

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
            history["diagnostics"]["train_action_abs_mean"].append(train_diag["action_abs_mean"])
            history["diagnostics"]["val_action_abs_mean"].append(val_diag["action_abs_mean"])
            history["diagnostics"]["train_delta_abs_mean"].append(train_diag["delta_abs_mean"])
            history["diagnostics"]["val_delta_abs_mean"].append(val_diag["delta_abs_mean"])
            history["diagnostics"]["train_pct_at_any_position_bound"].append(train_diag["pct_at_any_position_bound"])
            history["diagnostics"]["val_pct_at_any_position_bound"].append(val_diag["pct_at_any_position_bound"])
            history["diagnostics"]["train_pct_paths_touch_any_position_bound"].append(train_diag["pct_paths_touch_any_position_bound"])
            history["diagnostics"]["val_pct_paths_touch_any_position_bound"].append(val_diag["pct_paths_touch_any_position_bound"])
            history["learning_rate"].append(float(self.optimizer.param_groups[0]["lr"]))
            selection_score = _selection_score(train_loss, val_loss, val_diag)
            history["selection_scores"]["val"].append(float(selection_score))

            if train_loss < history["best_loss"]:
                history["best_loss"] = train_loss
            if selection_score < history["best_score"]:
                history["best_score"] = float(selection_score)
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
                    f"| val {val_loss:.4f} | select {selection_score:.4f}"
                )

        if self.best_state_dict is not None:
            self.gym.load_state_dict(self.best_state_dict)

        if history["best_epoch"] >= 0:
            k = int(history["best_epoch"])
            history["best_val_action_abs_mean"] = float(history["diagnostics"]["val_action_abs_mean"][k])
            history["best_val_delta_abs_mean"] = float(history["diagnostics"]["val_delta_abs_mean"][k])
            history["best_val_pct_at_any_position_bound"] = float(history["diagnostics"]["val_pct_at_any_position_bound"][k])
            history["best_val_pct_paths_touch_any_position_bound"] = float(
                history["diagnostics"]["val_pct_paths_touch_any_position_bound"][k]
            )
        else:
            history["best_val_action_abs_mean"] = float(init_val_diag["action_abs_mean"])
            history["best_val_delta_abs_mean"] = float(init_val_diag["delta_abs_mean"])
            history["best_val_pct_at_any_position_bound"] = float(init_val_diag["pct_at_any_position_bound"])
            history["best_val_pct_paths_touch_any_position_bound"] = float(
                init_val_diag["pct_paths_touch_any_position_bound"]
            )

        return history
