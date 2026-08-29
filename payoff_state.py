"""Observable payoff-state helpers for path-dependent real-data liabilities."""

from __future__ import annotations

import numpy as np


ASIAN_PAYOFF_STATE_VERSION = "asian-running-average-moneyness-end-fixings-v2"
ASIAN_PAYOFF_FEATURE = "asian_moneyness"
BASE_MODEL_FEATURES = ("price", "delta", "time_left")
ASIAN_MODEL_FEATURES = (*BASE_MODEL_FEATURES, ASIAN_PAYOFF_FEATURE)
ASIAN_LIABILITY_TYPES = frozenset({"asian", "asian_call", "asian_short_call"})


def is_asian_liability(liability_type):
    return str(liability_type).lower() in ASIAN_LIABILITY_TYPES


def resolve_asian_window(n_steps, start_step=0, end_step=None):
    """Return the validated half-open fixing window ``[start, end)``."""
    n_steps = int(n_steps)
    start = max(0, int(start_step))
    end = n_steps if end_step in (None, "") else min(n_steps, int(end_step))
    if start >= n_steps or end <= start:
        raise ValueError(
            f"Invalid Asian averaging window start={start}, end={end}, "
            f"n_steps={n_steps}"
        )
    return start, end


def arithmetic_running_average(spot, start_step=0, end_step=None):
    """Compute the observable running average for the configured fixing window.

    Before the first fixing, the value is zero. After the final fixing, the
    completed average is carried forward. The fixing count is deterministic
    from time and the configured window, so ``time_left`` disambiguates these
    phases for the policy.
    """
    spot = np.asarray(spot)
    if spot.ndim != 2:
        raise ValueError(f"Expected spot shape [paths, steps], got {spot.shape}")

    n_steps = int(spot.shape[1])
    start, end = resolve_asian_window(
        n_steps,
        start_step=start_step,
        end_step=end_step,
    )
    fixing_mask = np.zeros((n_steps,), dtype=spot.dtype)
    fixing_mask[start:end] = 1.0
    fixing_count = np.cumsum(fixing_mask)
    running_sum = np.cumsum(spot * fixing_mask[np.newaxis, :], axis=1)
    running_average = np.divide(
        running_sum,
        fixing_count[np.newaxis, :],
        out=np.zeros_like(running_sum),
        where=fixing_count[np.newaxis, :] > 0,
    )
    fixing_fraction = fixing_count / float(end - start)
    return (
        running_average.astype(spot.dtype, copy=False),
        fixing_count.astype(spot.dtype, copy=False),
        fixing_fraction.astype(spot.dtype, copy=False),
    )


def model_features_for_liability(liability_type):
    if is_asian_liability(liability_type):
        return list(ASIAN_MODEL_FEATURES)
    return list(BASE_MODEL_FEATURES)


def payoff_state_version_for_liability(liability_type):
    if is_asian_liability(liability_type):
        return ASIAN_PAYOFF_STATE_VERSION
    return None


def validate_model_features_for_liability(liability_type, feature_names):
    """Validate that a policy observes every state needed by its liability."""
    features = sorted(str(name) for name in feature_names)
    if is_asian_liability(liability_type) and ASIAN_PAYOFF_FEATURE not in features:
        raise ValueError(
            "Asian-option policies must include "
            f"'{ASIAN_PAYOFF_FEATURE}' so the running payoff state is observable. "
            f"Received features: {features}"
        )
    return features
