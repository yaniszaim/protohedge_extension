"""Utilities for cleaning and packaging multi-ticker WRDS option datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FEATURE_ORDER = ["spot", "call_price", "call_delta", "call_vega", "ivol"]
TEMPORAL_SPLIT_VERSION = "chronological-prewindow-v1"
OPTION_PATH_VERSION = "fixed-optionid-v1"
EPISODE_TIMING_VERSION = "decision-intervals-terminal-observation-v1"
DEFAULT_TRAIN_END_DATE = "2018-12-31"
DEFAULT_VALIDATION_END_DATE = "2021-12-31"


def discover_raw_pairs(raw_dir):
    """Return a per-ticker mapping of spot/options CSV paths under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    rows = {}
    for path in sorted(raw_dir.glob("*.csv")):
        stem = path.stem
        lower = stem.lower()
        if lower.endswith("_options"):
            ticker = stem[: -len("_options")].upper()
            rows.setdefault(ticker, {})["options_path"] = path
        elif lower.endswith("_crsp"):
            ticker = stem[: -len("_crsp")].upper()
            rows.setdefault(ticker, {})["spot_path"] = path
        elif lower.endswith("_srcp"):
            ticker = stem[: -len("_srcp")].upper()
            rows.setdefault(ticker, {})["spot_path"] = path
        else:
            rows.setdefault(stem.upper(), {})["other_path"] = path
    out = []
    for ticker, item in sorted(rows.items()):
        out.append(
            {
                "ticker": ticker,
                "spot_path": str(item.get("spot_path", "")),
                "options_path": str(item.get("options_path", "")),
                "complete_pair": bool(item.get("spot_path")) and bool(item.get("options_path")),
            }
        )
    return pd.DataFrame(out)


def _pick_existing_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Could not find any of {candidates} in columns {list(columns)}")


def clean_spot_csv(path, start_date=None, end_date=None):
    """Clean a CRSP spot CSV into ``date, spot`` format."""
    path = Path(path)
    df = pd.read_csv(path)

    date_col = _pick_existing_column(df.columns, ["date", "DlyCalDt"])
    price_col = _pick_existing_column(df.columns, ["PRC", "DlyPrc", "dlyprc"])

    out = df[[date_col, price_col]].copy()
    out.columns = ["date", "spot"]
    out["date"] = pd.to_datetime(out["date"])
    out["spot"] = pd.to_numeric(out["spot"], errors="coerce").abs()

    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.Timestamp(end_date)]

    out = out.dropna()
    out = out[np.isfinite(out["spot"])]
    out = out[out["spot"] > 0]
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def clean_option_csv(
    path,
    start_date=None,
    end_date=None,
    cp_flag="C",
    exercise_style="A",
    min_dte=1,
    max_dte=40,
    chunksize=250_000,
):
    """Clean tradable OptionMetrics quotes while retaining contract identifiers."""
    path = Path(path)
    required = [
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "impl_volatility",
        "delta",
        "vega",
        "optionid",
    ]
    optional = ["volume", "open_interest", "issuer", "exercise_style", "index_flag"]

    probe = pd.read_csv(path, nrows=2)
    keep_cols = []
    for col in required + optional:
        try:
            keep_cols.append(_pick_existing_column(probe.columns, [col]))
        except KeyError:
            if col in required:
                raise

    parts = []
    for chunk in pd.read_csv(path, usecols=keep_cols, chunksize=chunksize):
        rename = {
            _pick_existing_column(chunk.columns, ["date"]): "date",
            _pick_existing_column(chunk.columns, ["exdate"]): "exdate",
            _pick_existing_column(chunk.columns, ["cp_flag"]): "cp_flag",
            _pick_existing_column(chunk.columns, ["strike_price"]): "strike_price",
            _pick_existing_column(chunk.columns, ["best_bid"]): "best_bid",
            _pick_existing_column(chunk.columns, ["best_offer"]): "best_offer",
            _pick_existing_column(chunk.columns, ["impl_volatility"]): "ivol",
            _pick_existing_column(chunk.columns, ["delta"]): "call_delta",
            _pick_existing_column(chunk.columns, ["vega"]): "call_vega",
            _pick_existing_column(chunk.columns, ["optionid"]): "optionid",
        }
        for opt_name in optional:
            try:
                original = _pick_existing_column(chunk.columns, [opt_name])
                rename[original] = opt_name
            except KeyError:
                continue

        chunk = chunk.rename(columns=rename)
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["exdate"] = pd.to_datetime(chunk["exdate"])

        if start_date is not None:
            chunk = chunk[chunk["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            chunk = chunk[chunk["date"] <= pd.Timestamp(end_date)]

        chunk = chunk[chunk["cp_flag"].astype(str).str.upper() == str(cp_flag).upper()]
        if "exercise_style" in chunk.columns:
            chunk = chunk[chunk["exercise_style"].astype(str).str.upper() == str(exercise_style).upper()]

        for col in [
            "strike_price",
            "best_bid",
            "best_offer",
            "ivol",
            "call_delta",
            "call_vega",
            "optionid",
            "volume",
            "open_interest",
        ]:
            if col not in chunk.columns:
                continue
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk["call_price"] = 0.5 * (chunk["best_bid"] + chunk["best_offer"])
        chunk["ttm_days"] = (chunk["exdate"] - chunk["date"]).dt.days
        chunk = chunk[chunk["ttm_days"] >= int(min_dte)]
        if max_dte is not None:
            chunk = chunk[chunk["ttm_days"] <= int(max_dte)]

        chunk = chunk.dropna(
            subset=[
                "optionid",
                "strike_price",
                "best_bid",
                "best_offer",
                "call_price",
                "ivol",
                "call_delta",
                "call_vega",
            ]
        )
        chunk = chunk[np.isfinite(chunk["call_price"])]
        chunk = chunk[np.isfinite(chunk["ivol"])]
        chunk = chunk[(chunk["best_bid"] > 0) & (chunk["best_offer"] >= chunk["best_bid"])]
        chunk = chunk[chunk["call_price"] > 0]
        chunk = chunk[chunk["ivol"] > 0]
        chunk["optionid"] = chunk["optionid"].astype(np.int64)
        for col in ["volume", "open_interest"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].fillna(0.0)

        keep = [
            "date",
            "exdate",
            "strike_price",
            "call_price",
            "call_delta",
            "call_vega",
            "ivol",
            "ttm_days",
        ]
        for col in ["best_bid", "best_offer", "volume", "open_interest", "optionid", "issuer", "exercise_style"]:
            if col in chunk.columns:
                keep.append(col)
        parts.append(chunk[keep].copy())

    if not parts:
        return pd.DataFrame(
            columns=[
                "date",
                "exdate",
                "strike_price",
                "call_price",
                "call_delta",
                "call_vega",
                "ivol",
                "ttm_days",
                "optionid",
            ]
        )
    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values(["date", "exdate", "strike_price"]).reset_index(drop=True)
    return out


def infer_strike_scale(merged):
    """Infer a scaling factor for OptionMetrics strike prices."""
    spot_med = float(np.nanmedian(np.asarray(merged["spot"], dtype=float)))
    strike_med = float(np.nanmedian(np.asarray(merged["strike_price"], dtype=float)))
    if not np.isfinite(spot_med) or not np.isfinite(strike_med) or spot_med <= 0:
        return 1.0

    for scale in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        scaled = strike_med / scale
        if 0.25 * spot_med <= scaled <= 4.0 * spot_med:
            return float(scale)
    if strike_med > 20.0 * spot_med:
        return 1000.0
    return 1.0


def build_daily_feature_panel(spot_df, option_df, atm_target_dte=30):
    """Build a daily diagnostic panel that must not be used as a hedge price path.

    This helper independently selects a contract on every date. It remains useful
    for descriptive chain diagnostics, but contract-consistent hedging episodes
    must be built with :func:`build_contract_consistent_episode_dataset`.
    """
    merged = option_df.merge(spot_df, on="date", how="inner")
    if merged.empty:
        raise ValueError("No overlapping dates between spot and option data")

    strike_scale = infer_strike_scale(merged)
    merged["strike"] = pd.to_numeric(merged["strike_price"], errors="coerce") / float(strike_scale)
    merged["atm_dist"] = (merged["strike"] - merged["spot"]).abs()
    merged["ttm_target_dist"] = (merged["ttm_days"] - int(atm_target_dte)).abs()
    if "best_offer" in merged.columns and "best_bid" in merged.columns:
        merged["bid_ask_spread"] = merged["best_offer"] - merged["best_bid"]
    else:
        merged["bid_ask_spread"] = np.nan

    sort_cols = ["date", "atm_dist", "ttm_target_dist", "bid_ask_spread"]
    ascending = [True, True, True, True]

    if "open_interest" in merged.columns:
        merged["open_interest"] = pd.to_numeric(merged["open_interest"], errors="coerce").fillna(-1.0)
        sort_cols.append("open_interest")
        ascending.append(False)
    if "volume" in merged.columns:
        merged["volume"] = pd.to_numeric(merged["volume"], errors="coerce").fillna(-1.0)
        sort_cols.append("volume")
        ascending.append(False)

    merged = merged.sort_values(sort_cols, ascending=ascending)
    chosen = merged.groupby("date", as_index=False).first()
    chosen = chosen.sort_values("date").reset_index(drop=True)

    out = chosen[["date", "spot", "call_price", "call_delta", "call_vega", "ivol"]].copy()
    out = out.dropna()
    for col in RAW_FEATURE_ORDER:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["spot"] > 0) & (out["call_price"] > 0) & (out["ivol"] > 0)]
    out = out.sort_values("date").reset_index(drop=True)
    return out, {"strike_scale": float(strike_scale), "n_merged_rows": int(len(merged))}


def _prepare_contract_panel(spot_df, option_df, max_relative_spread=1.0):
    """Align valid quotes to spot dates and compute same-contract run lengths."""
    spot = spot_df[["date", "spot"]].copy()
    spot["date"] = pd.to_datetime(spot["date"])
    spot = spot.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    spot["_spot_row_index"] = np.arange(len(spot), dtype=np.int64)

    required = {
        "date",
        "exdate",
        "optionid",
        "strike_price",
        "best_bid",
        "best_offer",
        "call_price",
        "call_delta",
        "call_vega",
        "ivol",
        "ttm_days",
    }
    missing = required.difference(option_df.columns)
    if missing:
        raise KeyError(f"Option data is missing contract-path columns: {sorted(missing)}")

    options = option_df.copy()
    options["date"] = pd.to_datetime(options["date"])
    options["exdate"] = pd.to_datetime(options["exdate"])
    merged = options.merge(spot, on="date", how="inner")
    if merged.empty:
        raise ValueError("No overlapping valid option quotes and spot dates")

    strike_scale = infer_strike_scale(merged)
    merged["strike"] = (
        pd.to_numeric(merged["strike_price"], errors="coerce") / float(strike_scale)
    )
    merged["bid_ask_spread"] = merged["best_offer"] - merged["best_bid"]
    merged["relative_spread"] = (
        merged["bid_ask_spread"] / merged["call_price"].clip(lower=1e-12)
    )
    if max_relative_spread is not None:
        merged = merged[
            merged["relative_spread"].le(float(max_relative_spread))
        ].copy()
    for col in ("volume", "open_interest"):
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # A duplicate optionid/date observation is resolved using the most executable quote.
    merged = merged.sort_values(
        [
            "optionid",
            "_spot_row_index",
            "bid_ask_spread",
            "open_interest",
            "volume",
        ],
        ascending=[True, True, True, False, False],
    )
    merged = merged.drop_duplicates(["optionid", "_spot_row_index"], keep="first")
    merged = merged.sort_values(["optionid", "_spot_row_index"]).reset_index(drop=True)

    new_segment = merged["optionid"].ne(merged["optionid"].shift())
    new_segment |= merged["_spot_row_index"].diff().ne(1)
    new_segment |= merged["strike"].ne(merged["strike"].shift())
    new_segment |= merged["exdate"].ne(merged["exdate"].shift())
    merged["_contract_segment"] = new_segment.cumsum()
    offset = merged.groupby("_contract_segment").cumcount()
    segment_size = merged.groupby("_contract_segment")["optionid"].transform("size")
    merged["_forward_contract_observations"] = (segment_size - offset).astype(np.int64)
    merged["_contract_row_position"] = np.arange(len(merged), dtype=np.int64)
    return spot, merged, float(strike_scale)


def validate_contract_consistent_metadata(
    episode_metadata,
    n_steps,
    contract_paths=None,
):
    """Validate that every episode follows one listed option through its horizon."""
    metadata = episode_metadata.copy()
    n_observations = int(n_steps) + 1
    required = {
        "episode_index",
        "option_path_version",
        "optionid",
        "option_exdate",
        "start_date",
        "end_date",
        "n_observations",
        "episode_timing_version",
        "contract_observation_count",
        "contract_unique_optionids",
        "contract_unique_exdates",
        "contract_unique_strikes",
        "contract_expires_after_episode",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Episode metadata is missing contract fields: {sorted(missing)}")
    if not metadata["option_path_version"].eq(OPTION_PATH_VERSION).all():
        raise ValueError(f"Every episode must use option path version {OPTION_PATH_VERSION!r}")
    if not metadata["episode_timing_version"].eq(EPISODE_TIMING_VERSION).all():
        raise ValueError(
            f"Every episode must use timing version {EPISODE_TIMING_VERSION!r}"
        )
    if not pd.to_numeric(metadata["n_observations"]).eq(n_observations).all():
        raise ValueError("Episode observation count is inconsistent with its decision horizon")
    if not pd.to_numeric(metadata["contract_observation_count"]).eq(n_observations).all():
        raise ValueError("At least one hedge-option path does not span the complete episode")
    if not pd.to_numeric(metadata["contract_unique_optionids"]).eq(1).all():
        raise ValueError("At least one episode switches optionid during the hedge horizon")
    if not pd.to_numeric(metadata["contract_unique_exdates"]).eq(1).all():
        raise ValueError("At least one hedge contract changes expiration within an episode")
    if not pd.to_numeric(metadata["contract_unique_strikes"]).eq(1).all():
        raise ValueError("At least one hedge contract changes strike within an episode")

    end_date = pd.to_datetime(metadata["end_date"])
    exdate = pd.to_datetime(metadata["option_exdate"])
    if not (exdate > end_date).all():
        raise ValueError("Every hedge option must expire strictly after its liability episode")

    expires_flag = metadata["contract_expires_after_episode"]
    if expires_flag.dtype != bool:
        expires_flag = expires_flag.astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    if expires_flag.isna().any() or not expires_flag.all():
        raise ValueError("Invalid contract_expires_after_episode metadata")

    if contract_paths is not None:
        paths = contract_paths.copy()
        path_required = {"episode_index", "step", "date", "optionid", "exdate"}
        missing = path_required.difference(paths.columns)
        if missing:
            raise KeyError(f"Contract-path audit data is missing columns: {sorted(missing)}")
        grouped = paths.groupby("episode_index", sort=False)
        if not grouped.size().eq(n_observations).all():
            raise ValueError("Contract-path audit rows do not cover every episode observation")
        if not grouped["optionid"].nunique().eq(1).all():
            raise ValueError("Contract-path audit detected an optionid switch")
        if not grouped["exdate"].nunique().eq(1).all():
            raise ValueError("Contract-path audit detected an expiration change")
        if "strike" in paths and not grouped["strike"].nunique().eq(1).all():
            raise ValueError("Contract-path audit detected a strike change")
        if not grouped["date"].nunique().eq(n_observations).all():
            raise ValueError("Contract-path audit detected duplicate or missing dates")

        expected = metadata.set_index("episode_index")["optionid"].astype(np.int64)
        observed = grouped["optionid"].first().astype(np.int64)
        if not observed.reindex(expected.index).eq(expected).all():
            raise ValueError("Contract-path optionids disagree with episode metadata")
    return metadata


def build_contract_consistent_episode_dataset(
    spot_df,
    option_df,
    n_steps=20,
    max_gap_days=4,
    train_end_date=None,
    validation_end_date=None,
    train_frac=0.70,
    val_frac=0.15,
    embargo_steps=None,
    selection_min_dte=20,
    selection_max_dte=40,
    atm_target_dte=30,
    min_initial_open_interest=1,
    max_relative_spread=1.0,
    max_abs_moneyness=0.10,
):
    """Build episodes whose option hedge is one fixed listed contract.

    Calendar periods are split before window construction. For each eligible
    spot window, a near-ATM option is selected on the first date and retained
    only if the same ``optionid`` has valid quotes at all ``n_steps + 1``
    observations and expires strictly after the liability horizon. Thus,
    ``n_steps`` always denotes hedge decisions/price intervals, not rows.
    """
    n_steps = int(n_steps)
    n_observations = n_steps + 1
    spot, contracts, strike_scale = _prepare_contract_panel(
        spot_df,
        option_df,
        max_relative_spread=max_relative_spread,
    )
    train_end, validation_end = _resolve_chronological_boundaries(
        spot,
        train_end_date=train_end_date,
        validation_end_date=validation_end_date,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    embargo_steps = n_steps if embargo_steps is None else int(embargo_steps)
    if embargo_steps < 0:
        raise ValueError("embargo_steps must be non-negative")

    raw_periods = {
        "train": spot[spot["date"] <= train_end].copy(),
        "val": spot[
            (spot["date"] > train_end) & (spot["date"] <= validation_end)
        ].copy(),
        "test": spot[spot["date"] > validation_end].copy(),
    }
    periods = {
        "train": raw_periods["train"],
        "val": raw_periods["val"].iloc[embargo_steps:].copy(),
        "test": raw_periods["test"].iloc[embargo_steps:].copy(),
    }

    initial_candidates = contracts[
        contracts["ttm_days"].between(
            int(selection_min_dte),
            int(selection_max_dte),
            inclusive="both",
        )
        & contracts["_forward_contract_observations"].ge(n_observations)
        & contracts["open_interest"].ge(float(min_initial_open_interest))
    ].copy()
    candidates_by_start = {
        int(index): frame
        for index, frame in initial_candidates.groupby("_spot_row_index", sort=False)
    }

    episode_values = []
    metadata_rows = []
    contract_path_parts = []
    split_indices = {}
    skipped_no_contract = {}

    for split_name in ("train", "val", "test"):
        period = periods[split_name].copy()
        gaps = period["date"].diff().dt.days.fillna(1)
        period["_block_id"] = (gaps > int(max_gap_days)).cumsum()
        split_start = len(episode_values)
        skipped_no_contract[split_name] = 0

        for local_block_id, (_, block) in enumerate(period.groupby("_block_id")):
            block = block.reset_index(drop=True)
            if len(block) < n_observations:
                continue
            for start in range(len(block) - n_observations + 1):
                spot_window = block.iloc[start : start + n_observations]
                start_row = int(spot_window["_spot_row_index"].iloc[0])
                end_date = pd.Timestamp(spot_window["date"].iloc[-1])
                candidates = candidates_by_start.get(start_row)
                if candidates is None:
                    skipped_no_contract[split_name] += 1
                    continue
                candidates = candidates[candidates["exdate"] > end_date].copy()
                if candidates.empty:
                    skipped_no_contract[split_name] += 1
                    continue

                initial_spot = float(spot_window["spot"].iloc[0])
                candidates["atm_dist"] = (candidates["strike"] - initial_spot).abs()
                candidates = candidates[
                    (candidates["atm_dist"] / initial_spot).le(
                        float(max_abs_moneyness)
                    )
                ].copy()
                if candidates.empty:
                    skipped_no_contract[split_name] += 1
                    continue
                candidates["ttm_target_dist"] = (
                    candidates["ttm_days"] - int(atm_target_dte)
                ).abs()
                candidates = candidates.sort_values(
                    [
                        "atm_dist",
                        "ttm_target_dist",
                        "bid_ask_spread",
                        "open_interest",
                        "volume",
                    ],
                    ascending=[True, True, True, False, False],
                )
                chosen = candidates.iloc[0]
                row_position = int(chosen["_contract_row_position"])
                contract_path = contracts.iloc[
                    row_position : row_position + n_observations
                ].copy()
                expected_rows = spot_window["_spot_row_index"].to_numpy(dtype=np.int64)
                observed_rows = contract_path["_spot_row_index"].to_numpy(dtype=np.int64)
                if (
                    len(contract_path) != n_observations
                    or contract_path["optionid"].nunique() != 1
                    or not np.array_equal(observed_rows, expected_rows)
                ):
                    raise RuntimeError("Internal contract-continuity invariant failed")

                episode_index = len(episode_values)
                values = contract_path[RAW_FEATURE_ORDER].to_numpy(dtype=np.float32)
                episode_values.append(values)

                optionid = int(chosen["optionid"])
                option_exdate = pd.Timestamp(chosen["exdate"])
                relative_spread = contract_path["relative_spread"].to_numpy(dtype=float)
                metadata_rows.append(
                    {
                        "episode_index": episode_index,
                        "split": split_name,
                        "split_episode_index": episode_index - split_start,
                        "block_id": int(local_block_id),
                        "start_date": pd.Timestamp(spot_window["date"].iloc[0]).date().isoformat(),
                        "end_date": end_date.date().isoformat(),
                        "start_feature_row": int(expected_rows[0]),
                        "end_feature_row": int(expected_rows[-1]),
                        "n_steps": n_steps,
                        "n_observations": n_observations,
                        "episode_timing_version": EPISODE_TIMING_VERSION,
                        "option_path_version": OPTION_PATH_VERSION,
                        "optionid": optionid,
                        "option_exdate": option_exdate.date().isoformat(),
                        "option_strike": float(chosen["strike"]),
                        "liability_strike": initial_spot,
                        "initial_dte_days": int(chosen["ttm_days"]),
                        "final_dte_days": int(contract_path["ttm_days"].iloc[-1]),
                        "initial_moneyness": float(chosen["strike"] / initial_spot - 1.0),
                        "initial_open_interest": float(chosen["open_interest"]),
                        "initial_volume": float(chosen["volume"]),
                        "min_best_bid": float(contract_path["best_bid"].min()),
                        "mean_bid_ask_spread": float(contract_path["bid_ask_spread"].mean()),
                        "max_bid_ask_spread": float(contract_path["bid_ask_spread"].max()),
                        "mean_relative_spread": float(np.mean(relative_spread)),
                        "max_relative_spread": float(np.max(relative_spread)),
                        "contract_observation_count": int(len(contract_path)),
                        "contract_unique_optionids": int(contract_path["optionid"].nunique()),
                        "contract_unique_exdates": int(contract_path["exdate"].nunique()),
                        "contract_unique_strikes": int(contract_path["strike"].nunique()),
                        "contract_expires_after_episode": bool(option_exdate > end_date),
                        "hedge_option_distinct_from_liability": True,
                    }
                )

                audit = contract_path[
                    [
                        "date",
                        "exdate",
                        "optionid",
                        "strike",
                        "best_bid",
                        "best_offer",
                        "bid_ask_spread",
                        "relative_spread",
                        "volume",
                        "open_interest",
                        "ttm_days",
                        *RAW_FEATURE_ORDER,
                    ]
                ].copy()
                audit.insert(0, "step", np.arange(n_observations, dtype=np.int64))
                audit.insert(0, "split", split_name)
                audit.insert(0, "episode_index", episode_index)
                contract_path_parts.append(audit)

        split_indices[split_name] = np.arange(
            split_start,
            len(episode_values),
            dtype=np.int64,
        )
        if len(split_indices[split_name]) == 0:
            raise ValueError(
                f"No contract-consistent episodes were available for split '{split_name}'"
            )

    episodes = np.asarray(episode_values, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)
    contract_paths = pd.concat(contract_path_parts, ignore_index=True)
    split_indices = validate_chronological_splits(
        split_indices,
        n_episodes=len(episodes),
        episode_metadata=metadata,
    )
    validate_contract_consistent_metadata(
        metadata,
        n_steps=n_steps,
        contract_paths=contract_paths,
    )

    split_info = {
        "split_version": TEMPORAL_SPLIT_VERSION,
        "option_path_version": OPTION_PATH_VERSION,
        "episode_timing_version": EPISODE_TIMING_VERSION,
        "train_end_date": train_end.date().isoformat(),
        "validation_end_date": validation_end.date().isoformat(),
        "embargo_steps": int(embargo_steps),
        "n_steps": n_steps,
        "n_observations": n_observations,
        "max_gap_days": int(max_gap_days),
        "selection_min_dte": int(selection_min_dte),
        "selection_max_dte": int(selection_max_dte),
        "atm_target_dte": int(atm_target_dte),
        "min_initial_open_interest": float(min_initial_open_interest),
        "max_relative_spread": (
            None if max_relative_spread is None else float(max_relative_spread)
        ),
        "max_abs_moneyness": float(max_abs_moneyness),
        "quote_filter": "positive_bid_non_crossed_offer",
        "n_train_episodes": int(len(split_indices["train"])),
        "n_val_episodes": int(len(split_indices["val"])),
        "n_test_episodes": int(len(split_indices["test"])),
        "skipped_train_no_continuous_contract": int(skipped_no_contract["train"]),
        "skipped_val_no_continuous_contract": int(skipped_no_contract["val"]),
        "skipped_test_no_continuous_contract": int(skipped_no_contract["test"]),
        "train_episode_start_date": metadata.loc[
            metadata["split"] == "train", "start_date"
        ].min(),
        "train_episode_end_date": metadata.loc[
            metadata["split"] == "train", "end_date"
        ].max(),
        "val_episode_start_date": metadata.loc[
            metadata["split"] == "val", "start_date"
        ].min(),
        "val_episode_end_date": metadata.loc[
            metadata["split"] == "val", "end_date"
        ].max(),
        "test_episode_start_date": metadata.loc[
            metadata["split"] == "test", "start_date"
        ].min(),
        "test_episode_end_date": metadata.loc[
            metadata["split"] == "test", "end_date"
        ].max(),
    }
    build_info = {
        "strike_scale": float(strike_scale),
        "n_aligned_contract_quotes": int(len(contracts)),
        "n_contracts": int(contracts["optionid"].nunique()),
        "n_contract_path_rows": int(len(contract_paths)),
    }
    return episodes, metadata, split_indices, split_info, contract_paths, build_info


def build_episode_tensor(
    feature_df,
    n_steps=20,
    max_gap_days=4,
    return_metadata=False,
    split=None,
):
    """Create rolling decision episodes within contiguous trading-date blocks.

    A new block starts whenever the gap between consecutive feature dates
    exceeds ``max_gap_days``. This prevents episodes from stitching together
    disjoint option cycles separated by large holes in the cleaned panel.
    Each episode has ``n_steps + 1`` observations for ``n_steps`` decisions.
    """
    if "date" not in feature_df.columns:
        raise KeyError("feature_df must contain a 'date' column")

    feature_df = feature_df.sort_values("date").reset_index(drop=True).copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    gaps = feature_df["date"].diff().dt.days.fillna(1)
    block_id = (gaps > int(max_gap_days)).cumsum()

    n_steps = int(n_steps)
    n_observations = n_steps + 1
    episodes = []
    metadata = []
    for local_block_id, (_, block) in enumerate(feature_df.groupby(block_id)):
        values = block[RAW_FEATURE_ORDER].to_numpy(dtype=np.float32)
        if len(values) < n_observations:
            continue
        dates = block["date"].to_numpy()
        source_rows = (
            block["_feature_row_index"].to_numpy(dtype=np.int64)
            if "_feature_row_index" in block.columns
            else block.index.to_numpy(dtype=np.int64)
        )
        for start in range(len(values) - n_observations + 1):
            end = start + n_observations
            episodes.append(values[start:end])
            metadata.append(
                {
                    "split": split,
                    "split_episode_index": len(metadata),
                    "block_id": int(local_block_id),
                    "start_date": pd.Timestamp(dates[start]).date().isoformat(),
                    "end_date": pd.Timestamp(dates[end - 1]).date().isoformat(),
                    "start_feature_row": int(source_rows[start]),
                    "end_feature_row": int(source_rows[end - 1]),
                    "n_steps": n_steps,
                    "n_observations": n_observations,
                    "episode_timing_version": EPISODE_TIMING_VERSION,
                }
            )

    if not episodes:
        raise ValueError(
            f"Need at least one contiguous block with at least {n_observations} rows; "
            f"got {len(feature_df)} total rows and max_gap_days={max_gap_days}"
        )
    episode_tensor = np.asarray(episodes, dtype=np.float32)
    if return_metadata:
        return episode_tensor, pd.DataFrame(metadata)
    return episode_tensor


def _resolve_chronological_boundaries(
    feature_df,
    train_end_date=None,
    validation_end_date=None,
    train_frac=0.70,
    val_frac=0.15,
):
    dates = pd.Series(pd.to_datetime(feature_df["date"]).sort_values().unique())
    if dates.empty:
        raise ValueError("Cannot split an empty feature panel")

    if (train_end_date is None) != (validation_end_date is None):
        raise ValueError("Provide both train_end_date and validation_end_date, or neither")

    if train_end_date is None:
        train_frac = float(train_frac)
        val_frac = float(val_frac)
        if train_frac <= 0.0 or val_frac <= 0.0 or train_frac + val_frac >= 1.0:
            raise ValueError("Require train_frac > 0, val_frac > 0, and train_frac + val_frac < 1")
        n_dates = len(dates)
        train_stop = max(1, int(np.floor(n_dates * train_frac)))
        val_stop = max(train_stop + 1, int(np.floor(n_dates * (train_frac + val_frac))))
        if val_stop >= n_dates:
            raise ValueError("Chronological fractions leave no test dates")
        train_end = pd.Timestamp(dates.iloc[train_stop - 1])
        validation_end = pd.Timestamp(dates.iloc[val_stop - 1])
    else:
        train_end = pd.Timestamp(train_end_date)
        validation_end = pd.Timestamp(validation_end_date)

    if train_end >= validation_end:
        raise ValueError("train_end_date must be earlier than validation_end_date")
    if train_end < dates.min() or validation_end >= dates.max():
        raise ValueError(
            "Chronological boundaries must leave non-empty train, validation, and test calendars"
        )
    return train_end, validation_end


def validate_chronological_splits(
    splits,
    n_episodes,
    episode_metadata=None,
    require_full_coverage=True,
):
    """Validate disjoint fixed split indices and their calendar separation."""
    expected = ("train", "val", "test")
    normalized = {}
    for name in expected:
        if name not in splits:
            raise KeyError(f"Missing chronological split '{name}'")
        indices = np.asarray(splits[name], dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"Split '{name}' must be a non-empty one-dimensional index array")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Split '{name}' contains duplicate episode indices")
        if indices.min() < 0 or indices.max() >= int(n_episodes):
            raise ValueError(f"Split '{name}' contains out-of-range episode indices")
        normalized[name] = indices

    split_sets = {name: set(values.tolist()) for name, values in normalized.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_sets[left].intersection(split_sets[right]):
            raise ValueError(f"Episode indices overlap between '{left}' and '{right}'")

    union = set().union(*split_sets.values())
    if require_full_coverage and union != set(range(int(n_episodes))):
        raise ValueError("Chronological split indices do not cover the complete episode tensor")

    if episode_metadata is not None:
        metadata = episode_metadata.copy()
        required = {"episode_index", "split", "start_date", "end_date"}
        missing = required.difference(metadata.columns)
        if missing:
            raise KeyError(f"Episode metadata is missing columns: {sorted(missing)}")
        if len(metadata) != int(n_episodes):
            raise ValueError("Episode metadata row count does not match the episode tensor")
        if metadata["episode_index"].duplicated().any():
            raise ValueError("Episode metadata contains duplicate episode_index values")

        metadata["start_date"] = pd.to_datetime(metadata["start_date"])
        metadata["end_date"] = pd.to_datetime(metadata["end_date"])
        metadata = metadata.set_index("episode_index", drop=False)
        for name, indices in normalized.items():
            rows = metadata.loc[indices]
            if not rows["split"].eq(name).all():
                raise ValueError(f"Episode metadata labels disagree with split '{name}'")

        train_end = metadata.loc[normalized["train"], "end_date"].max()
        val_start = metadata.loc[normalized["val"], "start_date"].min()
        val_end = metadata.loc[normalized["val"], "end_date"].max()
        test_start = metadata.loc[normalized["test"], "start_date"].min()
        if train_end >= val_start:
            raise ValueError("Training and validation episode calendars overlap")
        if val_end >= test_start:
            raise ValueError("Validation and test episode calendars overlap")
    return normalized


def build_chronological_episode_dataset(
    feature_df,
    n_steps=20,
    max_gap_days=4,
    train_end_date=None,
    validation_end_date=None,
    train_frac=0.70,
    val_frac=0.15,
    embargo_steps=None,
):
    """Split dates first, then build rolling windows independently per period."""
    if "date" not in feature_df.columns:
        raise KeyError("feature_df must contain a 'date' column")

    features = feature_df.sort_values("date").reset_index(drop=True).copy()
    features["date"] = pd.to_datetime(features["date"])
    features["_feature_row_index"] = np.arange(len(features), dtype=np.int64)
    train_end, validation_end = _resolve_chronological_boundaries(
        features,
        train_end_date=train_end_date,
        validation_end_date=validation_end_date,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    n_steps = int(n_steps)
    n_observations = n_steps + 1
    embargo_steps = n_steps if embargo_steps is None else int(embargo_steps)
    if embargo_steps < 0:
        raise ValueError("embargo_steps must be non-negative")

    raw_periods = {
        "train": features[features["date"] <= train_end].copy(),
        "val": features[
            (features["date"] > train_end) & (features["date"] <= validation_end)
        ].copy(),
        "test": features[features["date"] > validation_end].copy(),
    }
    periods = {
        "train": raw_periods["train"],
        "val": raw_periods["val"].iloc[embargo_steps:].copy(),
        "test": raw_periods["test"].iloc[embargo_steps:].copy(),
    }

    episode_parts = []
    metadata_parts = []
    split_indices = {}
    next_episode_index = 0
    for split_name in ("train", "val", "test"):
        period = periods[split_name]
        if len(period) < n_observations:
            raise ValueError(
                f"Split '{split_name}' has only {len(period)} usable rows after embargo; "
                f"need at least {n_observations}"
            )
        split_episodes, split_metadata = build_episode_tensor(
            period,
            n_steps=n_steps,
            max_gap_days=max_gap_days,
            return_metadata=True,
            split=split_name,
        )
        indices = np.arange(
            next_episode_index,
            next_episode_index + len(split_episodes),
            dtype=np.int64,
        )
        split_metadata["episode_index"] = indices
        split_metadata["split_episode_index"] = np.arange(
            len(split_metadata), dtype=np.int64
        )
        split_indices[split_name] = indices
        episode_parts.append(split_episodes)
        metadata_parts.append(split_metadata)
        next_episode_index += len(split_episodes)

    episodes = np.concatenate(episode_parts, axis=0).astype(np.float32, copy=False)
    metadata = pd.concat(metadata_parts, axis=0, ignore_index=True)
    split_indices = validate_chronological_splits(
        split_indices,
        n_episodes=len(episodes),
        episode_metadata=metadata,
    )

    split_info = {
        "split_version": TEMPORAL_SPLIT_VERSION,
        "episode_timing_version": EPISODE_TIMING_VERSION,
        "train_end_date": train_end.date().isoformat(),
        "validation_end_date": validation_end.date().isoformat(),
        "embargo_steps": int(embargo_steps),
        "n_steps": n_steps,
        "n_observations": n_observations,
        "max_gap_days": int(max_gap_days),
        "n_train_episodes": int(len(split_indices["train"])),
        "n_val_episodes": int(len(split_indices["val"])),
        "n_test_episodes": int(len(split_indices["test"])),
        "train_episode_start_date": metadata.loc[
            metadata["split"] == "train", "start_date"
        ].min(),
        "train_episode_end_date": metadata.loc[
            metadata["split"] == "train", "end_date"
        ].max(),
        "val_episode_start_date": metadata.loc[
            metadata["split"] == "val", "start_date"
        ].min(),
        "val_episode_end_date": metadata.loc[
            metadata["split"] == "val", "end_date"
        ].max(),
        "test_episode_start_date": metadata.loc[
            metadata["split"] == "test", "start_date"
        ].min(),
        "test_episode_end_date": metadata.loc[
            metadata["split"] == "test", "end_date"
        ].max(),
    }
    return episodes, metadata, split_indices, split_info


def infer_episode_artifact_paths(data_path):
    data_path = Path(data_path)
    suffix = "_training_paths"
    ticker = data_path.stem[: -len(suffix)] if data_path.stem.endswith(suffix) else data_path.stem
    split_path = data_path.parent / f"{ticker}_chronological_splits.npz"
    metadata_path = data_path.parent.parent / "episode_metadata" / f"{ticker}_episode_metadata.csv"
    return split_path, metadata_path


def _scalar_from_npz(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value.tolist()


def _subsample_fixed_splits(splits, samples):
    if samples is None:
        return {name: np.asarray(values, dtype=np.int64) for name, values in splits.items()}
    samples = int(samples)
    if samples < 3:
        raise ValueError("samples must leave at least one episode in each split")
    total = sum(len(values) for values in splits.values())
    if samples > total:
        raise ValueError(f"samples={samples} exceeds available episodes={total}")

    names = ("train", "val", "test")
    exact = {name: samples * len(splits[name]) / total for name in names}
    counts = {name: max(1, min(len(splits[name]), int(np.floor(exact[name])))) for name in names}
    while sum(counts.values()) < samples:
        candidates = [name for name in names if counts[name] < len(splits[name])]
        name = max(candidates, key=lambda item: exact[item] - counts[item])
        counts[name] += 1
    while sum(counts.values()) > samples:
        candidates = [name for name in names if counts[name] > 1]
        name = min(candidates, key=lambda item: exact[item] - counts[item])
        counts[name] -= 1

    selected = {}
    for name in names:
        values = np.asarray(splits[name], dtype=np.int64)
        positions = np.linspace(0, len(values) - 1, counts[name], dtype=np.int64)
        selected[name] = values[positions]
    return selected


def load_chronological_splits(
    data_path,
    split_path=None,
    episode_metadata_path=None,
    samples=None,
):
    """Load and validate persisted chronological episode membership."""
    data_path = Path(data_path)
    inferred_split, inferred_metadata = infer_episode_artifact_paths(data_path)
    split_path = inferred_split if split_path is None else Path(split_path)
    episode_metadata_path = (
        inferred_metadata if episode_metadata_path is None else Path(episode_metadata_path)
    )
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing chronological split file '{split_path}'. Rebuild episodes from dated "
            "features before running a real-data experiment."
        )
    if not episode_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing episode-date metadata '{episode_metadata_path}'. Random historical "
            "splits are disabled because overlapping windows would leak across holdouts."
        )

    episode_tensor = np.load(data_path, mmap_mode="r")
    n_episodes = int(episode_tensor.shape[0])
    metadata = pd.read_csv(episode_metadata_path)
    with np.load(split_path, allow_pickle=False) as payload:
        splits = {name: np.asarray(payload[name], dtype=np.int64) for name in ("train", "val", "test")}
        split_info = {
            key: _scalar_from_npz(payload[key])
            for key in payload.files
            if key not in splits
        }
    if split_info.get("split_version") != TEMPORAL_SPLIT_VERSION:
        raise ValueError(
            f"Unsupported or missing temporal split version in '{split_path}'; "
            f"expected {TEMPORAL_SPLIT_VERSION!r}"
        )
    if split_info.get("option_path_version") != OPTION_PATH_VERSION:
        raise ValueError(
            f"Unsupported or missing hedge-option path version in '{split_path}'; "
            f"expected {OPTION_PATH_VERSION!r}. Rebuild episodes from raw OptionMetrics "
            "quotes so each path follows one optionid."
        )
    if split_info.get("episode_timing_version") != EPISODE_TIMING_VERSION:
        raise ValueError(
            f"Unsupported or missing episode timing version in '{split_path}'; "
            f"expected {EPISODE_TIMING_VERSION!r}. Rebuild episodes with one "
            "terminal observation after the final hedge decision."
        )
    n_steps = int(split_info.get("n_steps", -1))
    n_observations = int(split_info.get("n_observations", -1))
    if n_steps <= 0 or n_observations != n_steps + 1:
        raise ValueError(
            "Chronological split metadata must define n_observations = n_steps + 1"
        )
    if episode_tensor.ndim != 3 or int(episode_tensor.shape[1]) != n_observations:
        raise ValueError(
            f"Episode tensor shape {episode_tensor.shape} is inconsistent with "
            f"the recorded {n_steps} decisions and {n_observations} observations"
        )
    splits = validate_chronological_splits(
        splits,
        n_episodes=n_episodes,
        episode_metadata=metadata,
    )
    validate_contract_consistent_metadata(
        metadata,
        n_steps=n_steps,
    )
    recorded_contract_path = split_info.get("contract_path_path")
    contract_path_path = Path(recorded_contract_path) if recorded_contract_path else None
    if contract_path_path is not None and not contract_path_path.exists():
        portable_contract_path = (
            split_path.parent.parent / "contract_paths" / contract_path_path.name
        )
        if portable_contract_path.exists():
            contract_path_path = portable_contract_path
    if contract_path_path is None or not contract_path_path.exists():
        raise FileNotFoundError(
            "Missing contract-path audit file recorded by the episode builder. "
            "Rebuild the real-data panel before training."
        )
    split_info["contract_path_path"] = str(contract_path_path.resolve())
    selected = _subsample_fixed_splits(splits, samples=samples)
    split_info.update(
        {
            "split_path": str(split_path.resolve()),
            "episode_metadata_path": str(episode_metadata_path.resolve()),
            "selected_samples": None if samples is None else int(samples),
            "selected_n_train": int(len(selected["train"])),
            "selected_n_val": int(len(selected["val"])),
            "selected_n_test": int(len(selected["test"])),
        }
    )
    return selected, split_info, metadata


def save_chronological_episode_artifacts(
    episode_path,
    split_path,
    episode_metadata_path,
    episodes,
    splits,
    metadata,
    split_info,
):
    episode_path = Path(episode_path)
    split_path = Path(split_path)
    episode_metadata_path = Path(episode_metadata_path)
    for path in (episode_path, split_path, episode_metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(episode_path, episodes)
    np.savez(
        split_path,
        train=np.asarray(splits["train"], dtype=np.int64),
        val=np.asarray(splits["val"], dtype=np.int64),
        test=np.asarray(splits["test"], dtype=np.int64),
        **split_info,
    )
    metadata.to_csv(episode_metadata_path, index=False)


def _write_panel_metadata_summaries(manifest, output_dir, settings=None):
    """Keep panel-level summaries synchronized with persisted episode artifacts."""
    output_dir = Path(output_dir)
    ok = manifest.loc[manifest["status"].astype(str).str.lower() == "ok"].copy()
    if ok.empty:
        return

    summary_path = output_dir / "panel_summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = {}
    if settings:
        summary.update(settings)

    def _sum_column(name):
        return int(pd.to_numeric(ok[name], errors="coerce").fillna(0).sum())

    summary.update(
        {
            "tickers_processed": ok["ticker"].astype(str).tolist(),
            "n_tickers_processed": int(len(ok)),
            "total_episodes": _sum_column("n_episodes"),
            "total_train_episodes": _sum_column("n_train_episodes"),
            "total_validation_episodes": _sum_column("n_val_episodes"),
            "total_test_episodes": _sum_column("n_test_episodes"),
            "split_version": str(ok["split_version"].iloc[0]),
            "option_path_version": str(ok["option_path_version"].iloc[0]),
            "episode_timing_version": str(ok["episode_timing_version"].iloc[0]),
            "n_steps": int(ok["n_steps"].iloc[0]),
            "n_observations": int(ok["n_observations"].iloc[0]),
            "train_end_date": str(ok["train_end_date"].iloc[0]),
            "validation_end_date": str(ok["validation_end_date"].iloc[0]),
            "embargo_steps": int(ok["embargo_steps"].iloc[0]),
            "actual_feature_start_date": str(ok["feature_start_date"].min()),
            "actual_feature_end_date": str(ok["feature_end_date"].max()),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    diagnostics = []
    for _, row in ok.sort_values("ticker").iterrows():
        features = pd.read_csv(row["feature_path"])
        episodes = np.load(row["episode_path"], mmap_mode="r")
        dates = pd.Series(
            pd.to_datetime(features["date"], errors="coerce").dropna().unique()
        ).sort_values()
        gaps = dates.diff().dt.days.dropna()
        metadata = pd.read_csv(row["episode_metadata_path"])
        diagnostics.append(
            {
                "ticker": str(row["ticker"]),
                "n_rows": int(len(features)),
                "episode_shape": str(tuple(episodes.shape)),
                "has_nan": bool(np.isnan(episodes).any()),
                "has_inf": bool(np.isinf(episodes).any()),
                "spot_min": float(features["spot"].min()),
                "spot_max": float(features["spot"].max()),
                "call_price_min": float(features["call_price"].min()),
                "call_price_max": float(features["call_price"].max()),
                "delta_min": float(features["call_delta"].min()),
                "delta_max": float(features["call_delta"].max()),
                "delta_median": float(features["call_delta"].median()),
                "vega_min": float(features["call_vega"].min()),
                "vega_max": float(features["call_vega"].max()),
                "ivol_min": float(features["ivol"].min()),
                "ivol_max": float(features["ivol"].max()),
                "n_gap_gt4": int((gaps > 4).sum()),
                "max_gap_days": int(gaps.max()) if len(gaps) else 0,
                "split_version": str(row["split_version"]),
                "option_path_version": str(row["option_path_version"]),
                "n_unique_hedge_contracts": int(metadata["optionid"].nunique()),
                "all_contracts_fixed_within_episode": bool(
                    metadata["contract_unique_optionids"].eq(1).all()
                ),
                "all_contracts_expire_after_episode": bool(
                    metadata["contract_expires_after_episode"].astype(bool).all()
                ),
                "n_train_episodes": int(row["n_train_episodes"]),
                "n_val_episodes": int(row["n_val_episodes"]),
                "n_test_episodes": int(row["n_test_episodes"]),
                "train_episode_end_date": str(row["train_episode_end_date"]),
                "val_episode_start_date": str(row["val_episode_start_date"]),
                "val_episode_end_date": str(row["val_episode_end_date"]),
                "test_episode_start_date": str(row["test_episode_start_date"]),
            }
        )
    pd.DataFrame(diagnostics).to_csv(output_dir / "_episode_diagnostics.csv", index=False)


def process_panel_dataset(
    raw_dir,
    output_dir,
    tickers=None,
    start_date=None,
    end_date=None,
    min_dte=20,
    max_dte=40,
    n_steps=20,
    max_gap_days=4,
    train_end_date=DEFAULT_TRAIN_END_DATE,
    validation_end_date=DEFAULT_VALIDATION_END_DATE,
    train_frac=0.70,
    val_frac=0.15,
    embargo_steps=None,
    atm_target_dte=30,
    min_initial_open_interest=1,
    max_relative_spread=1.0,
    max_abs_moneyness=0.10,
):
    """
    Process raw ticker pairs into contract-consistent historical episodes.

    Returns a manifest DataFrame with one row per processed ticker.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    spot_dir = output_dir / "clean_spot"
    contract_path_dir = output_dir / "contract_paths"
    episode_dir = output_dir / "episodes"
    episode_metadata_dir = output_dir / "episode_metadata"
    for folder in [
        output_dir,
        spot_dir,
        contract_path_dir,
        episode_dir,
        episode_metadata_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    discovered = discover_raw_pairs(raw_dir)
    if tickers is not None:
        wanted = {str(t).upper() for t in tickers}
        discovered = discovered[discovered["ticker"].isin(wanted)].copy()

    rows = []
    for _, item in discovered.iterrows():
        ticker = str(item["ticker"]).upper()
        print(f"[panel] {ticker}: building fixed-optionid episodes")
        if not bool(item["complete_pair"]):
            rows.append(
                {
                    "ticker": ticker,
                    "status": "missing_pair",
                    "spot_path": item.get("spot_path", ""),
                    "options_path": item.get("options_path", ""),
                }
            )
            continue

        spot = clean_spot_csv(item["spot_path"], start_date=start_date, end_date=end_date)
        options = clean_option_csv(
            item["options_path"],
            start_date=start_date,
            end_date=end_date,
            min_dte=1,
            max_dte=max_dte,
        )
        if spot.empty or options.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "status": "empty_after_cleaning",
                    "spot_path": item["spot_path"],
                    "options_path": item["options_path"],
                    "n_spot_rows": int(len(spot)),
                    "n_option_rows": int(len(options)),
                }
            )
            continue

        (
            episodes,
            episode_metadata,
            splits,
            split_info,
            contract_paths,
            build_info,
        ) = build_contract_consistent_episode_dataset(
            spot,
            options,
            n_steps=n_steps,
            max_gap_days=max_gap_days,
            train_end_date=train_end_date,
            validation_end_date=validation_end_date,
            train_frac=train_frac,
            val_frac=val_frac,
            embargo_steps=embargo_steps,
            selection_min_dte=min_dte,
            selection_max_dte=max_dte,
            atm_target_dte=atm_target_dte,
            min_initial_open_interest=min_initial_open_interest,
            max_relative_spread=max_relative_spread,
            max_abs_moneyness=max_abs_moneyness,
        )
        gap_days = spot["date"].diff().dt.days.dropna()
        n_gap_breaks = int((gap_days > int(max_gap_days)).sum()) if len(gap_days) else 0
        max_observed_gap_days = int(gap_days.max()) if len(gap_days) else 0

        clean_spot_path = spot_dir / f"{ticker}_clean_spot.csv"
        contract_path_path = contract_path_dir / f"{ticker}_contract_paths.csv"
        episode_path = episode_dir / f"{ticker}_training_paths.npy"
        split_path = episode_dir / f"{ticker}_chronological_splits.npz"
        episode_metadata_path = episode_metadata_dir / f"{ticker}_episode_metadata.csv"
        split_info["contract_path_path"] = str(contract_path_path.resolve())
        spot.to_csv(clean_spot_path, index=False)
        contract_paths.to_csv(contract_path_path, index=False)
        save_chronological_episode_artifacts(
            episode_path=episode_path,
            split_path=split_path,
            episode_metadata_path=episode_metadata_path,
            episodes=episodes,
            splits=splits,
            metadata=episode_metadata,
            split_info=split_info,
        )

        rows.append(
            {
                "ticker": ticker,
                "status": "ok",
                "spot_path": str(Path(item["spot_path"]).resolve()),
                "options_path": str(Path(item["options_path"]).resolve()),
                "clean_spot_path": str(clean_spot_path.resolve()),
                # Kept for notebook compatibility; rows are explicitly episode-step long.
                "feature_path": str(contract_path_path.resolve()),
                "feature_layout": "episode-step-long",
                "contract_path_path": str(contract_path_path.resolve()),
                "episode_path": str(episode_path.resolve()),
                "split_path": str(split_path.resolve()),
                "episode_metadata_path": str(episode_metadata_path.resolve()),
                "n_spot_rows": int(len(spot)),
                "n_option_rows": int(len(options)),
                "n_feature_rows": int(len(contract_paths)),
                "n_contract_path_rows": int(len(contract_paths)),
                "n_episodes": int(len(episodes)),
                **split_info,
                **build_info,
                "feature_start_date": str(pd.to_datetime(contract_paths["date"]).min().date()),
                "feature_end_date": str(pd.to_datetime(contract_paths["date"]).max().date()),
                "n_steps": int(n_steps),
                "min_dte": int(min_dte),
                "max_dte": int(max_dte),
                "atm_target_dte": int(atm_target_dte),
                "min_initial_open_interest": float(min_initial_open_interest),
                "max_relative_spread": (
                    None if max_relative_spread is None else float(max_relative_spread)
                ),
                "max_abs_moneyness": float(max_abs_moneyness),
                "max_gap_days": int(max_gap_days),
                "n_gap_breaks": n_gap_breaks,
                "max_observed_gap_days": max_observed_gap_days,
            }
        )
        print(
            f"[panel] {ticker}: episodes={len(episodes)} "
            f"(train={len(splits['train'])}, val={len(splits['val'])}, "
            f"test={len(splits['test'])})"
        )

    manifest = pd.DataFrame(rows).sort_values(["status", "ticker"]).reset_index(drop=True)
    manifest.to_csv(output_dir / "panel_manifest.csv", index=False)
    _write_panel_metadata_summaries(
        manifest,
        output_dir,
        settings={
            "start_date": None if start_date is None else str(start_date),
            "end_date": None if end_date is None else str(end_date),
            "min_dte": int(min_dte),
            "max_dte": int(max_dte),
            "atm_target_dte": int(atm_target_dte),
            "min_initial_open_interest": float(min_initial_open_interest),
            "max_relative_spread": (
                None if max_relative_spread is None else float(max_relative_spread)
            ),
            "max_abs_moneyness": float(max_abs_moneyness),
            "n_steps": int(n_steps),
            "max_gap_days": int(max_gap_days),
        },
    )
    return manifest


def rebuild_panel_episode_artifacts(
    panel_dir,
    train_end_date=DEFAULT_TRAIN_END_DATE,
    validation_end_date=DEFAULT_VALIDATION_END_DATE,
    n_steps=20,
    max_gap_days=4,
    embargo_steps=None,
):
    """Rebuild contract-consistent artifacts from the original raw files."""
    panel_dir = Path(panel_dir)
    manifest_path = panel_dir / "panel_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing panel manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    ok = manifest[manifest["status"].astype(str).str.lower() == "ok"]
    if ok.empty:
        raise ValueError("Panel manifest has no successfully processed tickers")
    raw_dirs = {str(Path(path).resolve().parent) for path in ok["options_path"]}
    if len(raw_dirs) != 1:
        raise ValueError("Expected all raw option files to share one directory")

    summary_path = panel_dir / "panel_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return process_panel_dataset(
        raw_dir=next(iter(raw_dirs)),
        output_dir=panel_dir,
        tickers=ok["ticker"].astype(str).tolist(),
        start_date=summary.get("start_date"),
        end_date=summary.get("end_date"),
        min_dte=int(summary.get("min_dte", 20)),
        max_dte=int(summary.get("max_dte", 40)),
        n_steps=n_steps,
        max_gap_days=max_gap_days,
        train_end_date=train_end_date,
        validation_end_date=validation_end_date,
        embargo_steps=embargo_steps,
        atm_target_dte=int(summary.get("atm_target_dte", 30)),
        min_initial_open_interest=float(
            summary.get("min_initial_open_interest", 1)
        ),
        max_relative_spread=summary.get("max_relative_spread", 1.0),
        max_abs_moneyness=float(summary.get("max_abs_moneyness", 0.10)),
    )
