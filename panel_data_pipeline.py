"""Utilities for cleaning and packaging multi-ticker WRDS option datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_FEATURE_ORDER = ["spot", "call_price", "call_delta", "call_vega", "ivol"]


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
    min_dte=20,
    max_dte=40,
    chunksize=250_000,
):
    """Clean an OptionMetrics CSV into daily call-chain rows for one underlying."""
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
    ]
    optional = ["volume", "open_interest", "optionid", "issuer", "exercise_style", "index_flag"]

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

        for col in ["strike_price", "best_bid", "best_offer", "ivol", "call_delta", "call_vega"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk["call_price"] = 0.5 * (chunk["best_bid"] + chunk["best_offer"])
        chunk["ttm_days"] = (chunk["exdate"] - chunk["date"]).dt.days
        chunk = chunk[(chunk["ttm_days"] > int(min_dte)) & (chunk["ttm_days"] < int(max_dte))]

        chunk = chunk.dropna(subset=["strike_price", "call_price", "ivol", "call_delta", "call_vega"])
        chunk = chunk[np.isfinite(chunk["call_price"])]
        chunk = chunk[np.isfinite(chunk["ivol"])]
        chunk = chunk[chunk["call_price"] > 0]
        chunk = chunk[chunk["ivol"] > 0]

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
    """Merge cleaned spot and options and select one near-ATM call per date."""
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


def build_episode_tensor(feature_df, n_steps=20, max_gap_days=4):
    """Create rolling episode windows within contiguous trading-date blocks.

    A new block starts whenever the gap between consecutive feature dates
    exceeds ``max_gap_days``. This prevents episodes from stitching together
    disjoint option cycles separated by large holes in the cleaned panel.
    """
    if "date" not in feature_df.columns:
        raise KeyError("feature_df must contain a 'date' column")

    feature_df = feature_df.sort_values("date").reset_index(drop=True).copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    gaps = feature_df["date"].diff().dt.days.fillna(1)
    block_id = (gaps > int(max_gap_days)).cumsum()

    episodes = []
    for _, block in feature_df.groupby(block_id):
        values = block[RAW_FEATURE_ORDER].to_numpy(dtype=np.float32)
        if len(values) < int(n_steps):
            continue
        episodes.extend(values[i : i + int(n_steps)] for i in range(len(values) - int(n_steps)))

    if not episodes:
        raise ValueError(
            f"Need at least one contiguous block with more than {n_steps} rows; "
            f"got {len(feature_df)} total rows and max_gap_days={max_gap_days}"
        )
    return np.asarray(episodes, dtype=np.float32)


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
):
    """
    Process all discovered raw ticker pairs into cleaned spot/features/episode files.

    Returns a manifest DataFrame with one row per processed ticker.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    spot_dir = output_dir / "clean_spot"
    feature_dir = output_dir / "features"
    episode_dir = output_dir / "episodes"
    for folder in [output_dir, spot_dir, feature_dir, episode_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    discovered = discover_raw_pairs(raw_dir)
    if tickers is not None:
        wanted = {str(t).upper() for t in tickers}
        discovered = discovered[discovered["ticker"].isin(wanted)].copy()

    rows = []
    for _, item in discovered.iterrows():
        ticker = str(item["ticker"]).upper()
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
            min_dte=min_dte,
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

        features, meta = build_daily_feature_panel(spot, options)
        episodes = build_episode_tensor(features, n_steps=n_steps, max_gap_days=max_gap_days)
        gap_days = features["date"].diff().dt.days.dropna()
        n_gap_breaks = int((gap_days > int(max_gap_days)).sum()) if len(gap_days) else 0
        max_observed_gap_days = int(gap_days.max()) if len(gap_days) else 0

        clean_spot_path = spot_dir / f"{ticker}_clean_spot.csv"
        feature_path = feature_dir / f"{ticker}_option_features.csv"
        episode_path = episode_dir / f"{ticker}_training_paths.npy"
        spot.to_csv(clean_spot_path, index=False)
        features.to_csv(feature_path, index=False)
        np.save(episode_path, episodes)

        rows.append(
            {
                "ticker": ticker,
                "status": "ok",
                "spot_path": str(Path(item["spot_path"]).resolve()),
                "options_path": str(Path(item["options_path"]).resolve()),
                "clean_spot_path": str(clean_spot_path.resolve()),
                "feature_path": str(feature_path.resolve()),
                "episode_path": str(episode_path.resolve()),
                "n_spot_rows": int(len(spot)),
                "n_option_rows": int(len(options)),
                "n_feature_rows": int(len(features)),
                "n_episodes": int(len(episodes)),
                "feature_start_date": str(features["date"].min().date()),
                "feature_end_date": str(features["date"].max().date()),
                "strike_scale": float(meta["strike_scale"]),
                "n_steps": int(n_steps),
                "min_dte": int(min_dte),
                "max_dte": int(max_dte),
                "max_gap_days": int(max_gap_days),
                "n_gap_breaks": n_gap_breaks,
                "max_observed_gap_days": max_observed_gap_days,
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["status", "ticker"]).reset_index(drop=True)
    manifest.to_csv(output_dir / "panel_manifest.csv", index=False)
    return manifest
