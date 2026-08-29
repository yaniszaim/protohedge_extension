"""Outcome semantics and dependent-data inference for real-data hedging tests."""

from __future__ import annotations

import numpy as np


OUTCOME_DEFINITION_VERSION = "liability-offset-excluding-premium-v1"
OUTCOME_NAME = "liability_offset"
PREMIUM_INCLUDED = False
DEFAULT_TAIL_PROBABILITY = 0.05
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
INFERENCE_VERSION = "paired-circular-block-seed-average-v2"


def as_finite_vector(values, name=OUTCOME_NAME):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def empirical_lower_tail_cvar(values, alpha=DEFAULT_TAIL_PROBABILITY):
    """Return empirical lower-tail VaR and CVaR for an outcome to maximize."""
    values = as_finite_vector(values)
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}")
    var = float(np.quantile(values, alpha))
    tail = values[values <= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


def summarize_liability_offset(values, alpha=DEFAULT_TAIL_PROBABILITY):
    """Summarize the premium-excluded residual offset of the short liability."""
    offset = as_finite_vector(values)
    var, cvar = empirical_lower_tail_cvar(offset, alpha=alpha)
    downside = np.minimum(offset, 0.0)
    return {
        "liability_offset_mean": float(offset.mean()),
        "liability_offset_std": float(offset.std()),
        "liability_offset_variance": float(offset.var()),
        "liability_offset_p01": float(np.quantile(offset, 0.01)),
        "liability_offset_p05": var,
        "liability_offset_cvar05": cvar,
        "liability_offset_p50": float(np.quantile(offset, 0.50)),
        "liability_offset_p95": float(np.quantile(offset, 0.95)),
        "liability_offset_mae": float(np.mean(np.abs(offset))),
        "liability_offset_rmse": float(np.sqrt(np.mean(offset ** 2))),
        "liability_offset_downside_deviation": float(
            np.sqrt(np.mean(downside ** 2))
        ),
        "negative_offset_rate": float(np.mean(offset < 0.0)),
        "premium_included": PREMIUM_INCLUDED,
        "outcome_definition": OUTCOME_DEFINITION_VERSION,
    }


def circular_block_bootstrap_indices(
    n_observations,
    block_length,
    n_bootstrap,
    random_state=0,
):
    """Generate circular-block samples that preserve local temporal dependence."""
    n_observations = int(n_observations)
    block_length = int(block_length)
    n_bootstrap = int(n_bootstrap)
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    block_length = min(block_length, n_observations)
    blocks_per_sample = int(np.ceil(n_observations / block_length))
    rng = np.random.default_rng(int(random_state))
    offsets = np.arange(block_length, dtype=np.int64)
    samples = np.empty((n_bootstrap, n_observations), dtype=np.int64)
    for bootstrap_idx in range(n_bootstrap):
        starts = rng.integers(
            0,
            n_observations,
            size=blocks_per_sample,
        )
        indices = (
            starts[:, np.newaxis] + offsets[np.newaxis, :]
        ) % n_observations
        samples[bootstrap_idx] = indices.reshape(-1)[:n_observations]
    return samples


def _paired_statistics(candidate_sample, benchmark_sample):
    _, candidate_cvar = empirical_lower_tail_cvar(candidate_sample)
    _, benchmark_cvar = empirical_lower_tail_cvar(benchmark_sample)
    candidate_rmse = np.sqrt(np.mean(candidate_sample ** 2))
    benchmark_rmse = np.sqrt(np.mean(benchmark_sample ** 2))
    candidate_downside = np.sqrt(
        np.mean(np.minimum(candidate_sample, 0.0) ** 2)
    )
    benchmark_downside = np.sqrt(
        np.mean(np.minimum(benchmark_sample, 0.0) ** 2)
    )
    return {
        "liability_offset_mean_difference": float(
            candidate_sample.mean() - benchmark_sample.mean()
        ),
        "liability_offset_cvar05_difference": float(
            candidate_cvar - benchmark_cvar
        ),
        "liability_offset_rmse_improvement": float(
            benchmark_rmse - candidate_rmse
        ),
        "liability_offset_downside_deviation_improvement": float(
            benchmark_downside - candidate_downside
        ),
    }


def as_finite_seed_matrix(values, name):
    """Return outcomes as ``[seed, chronological observation]``."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(
            f"{name} must have shape [seed, observation], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _seed_averaged_paired_statistics(candidate_matrix, benchmark_matrix):
    per_seed = [
        _paired_statistics(candidate_matrix[i], benchmark_matrix[i])
        for i in range(candidate_matrix.shape[0])
    ]
    return {
        metric: float(np.mean([row[metric] for row in per_seed]))
        for metric in per_seed[0]
    }


def paired_seed_circular_block_bootstrap(
    candidate_by_seed,
    benchmark_by_seed,
    block_length,
    n_bootstrap=2000,
    confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
    random_state=0,
):
    """Paired temporal intervals for the table's seed-averaged estimator."""
    candidate = as_finite_seed_matrix(candidate_by_seed, "candidate_by_seed")
    benchmark = as_finite_seed_matrix(benchmark_by_seed, "benchmark_by_seed")
    if candidate.shape != benchmark.shape:
        raise ValueError(
            "Seed-aware paired inputs must have identical shapes, got "
            f"{candidate.shape} and {benchmark.shape}"
        )
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    estimates = _seed_averaged_paired_statistics(candidate, benchmark)
    samples = circular_block_bootstrap_indices(
        n_observations=candidate.shape[1],
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    draws = {
        metric: np.empty((int(n_bootstrap),), dtype=np.float64)
        for metric in estimates
    }
    for draw_idx, indices in enumerate(samples):
        values = _seed_averaged_paired_statistics(
            candidate[:, indices],
            benchmark[:, indices],
        )
        for metric, value in values.items():
            draws[metric][draw_idx] = value

    alpha = 1.0 - confidence
    return [
        {
            "metric": metric,
            "estimate": float(estimate),
            "ci_low": float(np.quantile(draws[metric], alpha / 2.0)),
            "ci_high": float(np.quantile(draws[metric], 1.0 - alpha / 2.0)),
            "confidence": confidence,
            "n_seeds": int(candidate.shape[0]),
            "n_observations": int(candidate.shape[1]),
            "block_length": int(min(block_length, candidate.shape[1])),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_method": "paired_circular_block_seed_average",
            "inference_version": INFERENCE_VERSION,
        }
        for metric, estimate in estimates.items()
    ]


def paired_panel_seed_circular_block_bootstrap(
    candidate_by_group_seed,
    benchmark_by_group_seed,
    block_length,
    n_bootstrap=2000,
    confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
    random_state=0,
):
    """Equal-ticker intervals matching seed-then-ticker table aggregation."""
    groups = sorted(set(candidate_by_group_seed))
    if set(groups) != set(benchmark_by_group_seed) or not groups:
        raise ValueError("Candidate and benchmark groups must be identical and non-empty")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    pairs = {}
    block_lengths = {}
    samples = {}
    for group_idx, group in enumerate(groups):
        candidate = as_finite_seed_matrix(
            candidate_by_group_seed[group], f"candidate[{group!r}]"
        )
        benchmark = as_finite_seed_matrix(
            benchmark_by_group_seed[group], f"benchmark[{group!r}]"
        )
        if candidate.shape != benchmark.shape:
            raise ValueError(f"Group {group!r} has mismatched shapes")
        pairs[group] = (candidate, benchmark)
        requested = int(block_length[group]) if isinstance(block_length, dict) else int(block_length)
        block_lengths[group] = min(requested, candidate.shape[1])
        samples[group] = circular_block_bootstrap_indices(
            candidate.shape[1],
            block_lengths[group],
            n_bootstrap,
            random_state=int(random_state) + 104729 * group_idx,
        )

    group_estimates = [
        _seed_averaged_paired_statistics(*pairs[group]) for group in groups
    ]
    metric_names = list(group_estimates[0])
    estimates = {
        metric: float(np.mean([row[metric] for row in group_estimates]))
        for metric in metric_names
    }
    draws = {
        metric: np.empty((n_bootstrap,), dtype=np.float64)
        for metric in metric_names
    }
    for draw_idx in range(n_bootstrap):
        group_values = []
        for group in groups:
            indices = samples[group][draw_idx]
            candidate, benchmark = pairs[group]
            group_values.append(
                _seed_averaged_paired_statistics(
                    candidate[:, indices], benchmark[:, indices]
                )
            )
        for metric in metric_names:
            draws[metric][draw_idx] = np.mean(
                [row[metric] for row in group_values]
            )

    alpha = 1.0 - confidence
    return [
        {
            "metric": metric,
            "estimate": estimate,
            "ci_low": float(np.quantile(draws[metric], alpha / 2.0)),
            "ci_high": float(np.quantile(draws[metric], 1.0 - alpha / 2.0)),
            "confidence": confidence,
            "n_groups": len(groups),
            "n_seeds": int(next(iter(pairs.values()))[0].shape[0]),
            "n_observations_total": int(sum(pair[0].shape[1] for pair in pairs.values())),
            "min_block_length": int(min(block_lengths.values())),
            "max_block_length": int(max(block_lengths.values())),
            "n_bootstrap": n_bootstrap,
            "bootstrap_method": "paired_circular_block_seed_then_equal_group",
            "aggregation": "seed_average_then_equal_group",
            "inference_version": INFERENCE_VERSION,
        }
        for metric, estimate in estimates.items()
    ]


def paired_circular_block_bootstrap(
    candidate_values,
    benchmark_values,
    block_length,
    n_bootstrap=2000,
    confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
    random_state=0,
):
    """Compare paired chronological outcomes using circular-block intervals.

    Positive estimates favor the candidate for every returned metric.
    """
    candidate = as_finite_vector(candidate_values, name="candidate_values")
    benchmark = as_finite_vector(benchmark_values, name="benchmark_values")
    if candidate.shape != benchmark.shape:
        raise ValueError(
            "Paired bootstrap inputs must have identical shapes, got "
            f"{candidate.shape} and {benchmark.shape}"
        )
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must lie strictly between zero and one, got {confidence}"
        )

    estimates = _paired_statistics(candidate, benchmark)
    samples = circular_block_bootstrap_indices(
        n_observations=len(candidate),
        block_length=block_length,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    bootstrap_values = {
        metric: np.empty((int(n_bootstrap),), dtype=np.float64)
        for metric in estimates
    }
    for bootstrap_idx, indices in enumerate(samples):
        values = _paired_statistics(
            candidate[indices],
            benchmark[indices],
        )
        for metric, value in values.items():
            bootstrap_values[metric][bootstrap_idx] = value

    alpha = 1.0 - confidence
    rows = []
    for metric, estimate in estimates.items():
        values = bootstrap_values[metric]
        rows.append(
            {
                "metric": metric,
                "estimate": float(estimate),
                "ci_low": float(np.quantile(values, alpha / 2.0)),
                "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
                "confidence": confidence,
                "n_observations": int(len(candidate)),
                "block_length": int(min(block_length, len(candidate))),
                "n_bootstrap": int(n_bootstrap),
                "bootstrap_method": "paired_circular_block",
            }
        )
    return rows


def paired_panel_circular_block_bootstrap(
    candidate_by_group,
    benchmark_by_group,
    block_length,
    n_bootstrap=2000,
    confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
    random_state=0,
):
    """Return equal-group paired intervals with within-group block resampling.

    Each group is resampled independently in chronological circular blocks.
    Statistics are calculated within group and then averaged with equal group
    weight, matching the paper's equal-ticker aggregation.
    """
    candidate_keys = set(candidate_by_group)
    benchmark_keys = set(benchmark_by_group)
    if candidate_keys != benchmark_keys:
        raise ValueError(
            "candidate_by_group and benchmark_by_group must have identical "
            f"keys, got {sorted(candidate_keys)} and {sorted(benchmark_keys)}"
        )
    if not candidate_keys:
        raise ValueError("At least one group is required")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must lie strictly between zero and one, got {confidence}"
        )
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    groups = sorted(candidate_keys)
    pairs = {}
    group_block_lengths = {}
    for group in groups:
        candidate = as_finite_vector(
            candidate_by_group[group],
            name=f"candidate_by_group[{group!r}]",
        )
        benchmark = as_finite_vector(
            benchmark_by_group[group],
            name=f"benchmark_by_group[{group!r}]",
        )
        if candidate.shape != benchmark.shape:
            raise ValueError(
                f"Group {group!r} has mismatched paired shapes "
                f"{candidate.shape} and {benchmark.shape}"
            )
        pairs[group] = (candidate, benchmark)
        if isinstance(block_length, dict):
            requested_length = int(block_length[group])
        else:
            requested_length = int(block_length)
        if requested_length <= 0:
            raise ValueError("All block lengths must be positive")
        group_block_lengths[group] = min(
            requested_length,
            len(candidate),
        )

    group_estimates = [
        _paired_statistics(*pairs[group])
        for group in groups
    ]
    metric_names = list(group_estimates[0])
    estimates = {
        metric: float(
            np.mean([values[metric] for values in group_estimates])
        )
        for metric in metric_names
    }

    bootstrap_indices = {}
    for group_idx, group in enumerate(groups):
        bootstrap_indices[group] = circular_block_bootstrap_indices(
            n_observations=len(pairs[group][0]),
            block_length=group_block_lengths[group],
            n_bootstrap=n_bootstrap,
            random_state=int(random_state) + 104729 * group_idx,
        )

    bootstrap_values = {
        metric: np.empty((n_bootstrap,), dtype=np.float64)
        for metric in metric_names
    }
    for bootstrap_idx in range(n_bootstrap):
        values_by_group = []
        for group in groups:
            indices = bootstrap_indices[group][bootstrap_idx]
            candidate, benchmark = pairs[group]
            values_by_group.append(
                _paired_statistics(
                    candidate[indices],
                    benchmark[indices],
                )
            )
        for metric in metric_names:
            bootstrap_values[metric][bootstrap_idx] = np.mean(
                [values[metric] for values in values_by_group]
            )

    alpha = 1.0 - confidence
    rows = []
    for metric, estimate in estimates.items():
        values = bootstrap_values[metric]
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_low": float(np.quantile(values, alpha / 2.0)),
                "ci_high": float(
                    np.quantile(values, 1.0 - alpha / 2.0)
                ),
                "confidence": confidence,
                "n_groups": len(groups),
                "n_observations_total": int(
                    sum(len(pairs[group][0]) for group in groups)
                ),
                "min_block_length": int(
                    min(group_block_lengths.values())
                ),
                "max_block_length": int(
                    max(group_block_lengths.values())
                ),
                "n_bootstrap": n_bootstrap,
                "bootstrap_method": (
                    "paired_circular_block_equal_group"
                ),
                "aggregation": "equal_weight_across_groups",
            }
        )
    return rows
