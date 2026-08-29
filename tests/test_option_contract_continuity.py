import unittest

import numpy as np
import pandas as pd

from deephedging.panel_data_pipeline import (
    EPISODE_TIMING_VERSION,
    OPTION_PATH_VERSION,
    build_contract_consistent_episode_dataset,
)


def _spot_and_options(n_rows=90):
    dates = pd.bdate_range("2020-01-02", periods=n_rows)
    spot = pd.DataFrame(
        {
            "date": dates,
            "spot": 100.0 + 0.1 * np.arange(n_rows),
        }
    )
    exdate = dates[-1] + pd.Timedelta(days=365)

    rows = []
    for step, row in spot.iterrows():
        stable_price = 5.0 + 0.05 * step
        rows.append(
            {
                "date": row["date"],
                "exdate": exdate,
                "optionid": 100,
                "strike_price": 100_000,
                "best_bid": stable_price - 0.05,
                "best_offer": stable_price + 0.05,
                "call_price": stable_price,
                "call_delta": 0.50,
                "call_vega": 10.0,
                "ivol": 0.20,
                "ttm_days": int((exdate - row["date"]).days),
                "volume": 10,
                "open_interest": 100,
            }
        )
        # This quote is closer to ATM but exists on only one date, so it is not tradable
        # as the cumulative option inventory carried through an episode.
        rows.append(
            {
                "date": row["date"],
                "exdate": exdate,
                "optionid": 10_000 + step,
                "strike_price": int(round(row["spot"] * 1000)),
                "best_bid": 98.95,
                "best_offer": 99.05,
                "call_price": 99.0,
                "call_delta": 0.50,
                "call_vega": 10.0,
                "ivol": 0.20,
                "ttm_days": int((exdate - row["date"]).days),
                "volume": 1000,
                "open_interest": 1000,
            }
        )
    return spot, pd.DataFrame(rows)


class OptionContractContinuityTests(unittest.TestCase):
    def test_episode_builder_rejects_nonpersistent_atm_decoys(self):
        spot, options = _spot_and_options()
        episodes, metadata, splits, info, paths, build_info = (
            build_contract_consistent_episode_dataset(
                spot,
                options,
                n_steps=5,
                train_end_date=spot.loc[29, "date"],
                validation_end_date=spot.loc[59, "date"],
                embargo_steps=4,
                selection_min_dte=1,
                selection_max_dte=1000,
                atm_target_dte=30,
            )
        )

        self.assertEqual(info["option_path_version"], OPTION_PATH_VERSION)
        self.assertEqual(info["episode_timing_version"], EPISODE_TIMING_VERSION)
        self.assertEqual(episodes.shape[1], 6)
        self.assertGreater(len(episodes), 0)
        self.assertTrue(metadata["optionid"].eq(100).all())
        self.assertTrue(metadata["contract_unique_optionids"].eq(1).all())
        self.assertTrue(metadata["contract_observation_count"].eq(6).all())
        self.assertTrue(metadata["contract_expires_after_episode"].all())
        self.assertTrue(
            paths.groupby("episode_index")["optionid"].nunique().eq(1).all()
        )
        self.assertTrue(paths["optionid"].eq(100).all())
        self.assertTrue((episodes[:, :, 1] < 20.0).all())
        self.assertEqual(build_info["n_contract_path_rows"], len(episodes) * 6)
        self.assertEqual(
            sum(len(indices) for indices in splits.values()),
            len(episodes),
        )

    def test_hedge_contract_expires_after_liability_horizon(self):
        spot, options = _spot_and_options()
        _, metadata, _, _, _, _ = build_contract_consistent_episode_dataset(
            spot,
            options,
            n_steps=5,
            train_end_date=spot.loc[29, "date"],
            validation_end_date=spot.loc[59, "date"],
            embargo_steps=4,
            selection_min_dte=1,
            selection_max_dte=1000,
        )
        self.assertTrue(
            (
                pd.to_datetime(metadata["option_exdate"])
                > pd.to_datetime(metadata["end_date"])
            ).all()
        )
        self.assertTrue(metadata["hedge_option_distinct_from_liability"].all())

    def test_same_optionid_strike_adjustment_breaks_contract_run(self):
        spot, options = _spot_and_options()
        adjustment_step = 15
        stable_mask = options["optionid"].eq(100)
        options.loc[
            stable_mask & (options["date"] >= spot.loc[adjustment_step, "date"]),
            "strike_price",
        ] = 50_000

        fallback_rows = []
        exdate = pd.Timestamp(options["exdate"].iloc[0])
        for step, row in spot.iterrows():
            price = 4.0 + 0.04 * step
            fallback_rows.append(
                {
                    "date": row["date"],
                    "exdate": exdate,
                    "optionid": 200,
                    "strike_price": 105_000,
                    "best_bid": price - 0.05,
                    "best_offer": price + 0.05,
                    "call_price": price,
                    "call_delta": 0.45,
                    "call_vega": 9.0,
                    "ivol": 0.22,
                    "ttm_days": int((exdate - row["date"]).days),
                    "volume": 20,
                    "open_interest": 200,
                }
            )
        options = pd.concat([options, pd.DataFrame(fallback_rows)], ignore_index=True)

        _, metadata, _, _, paths, _ = build_contract_consistent_episode_dataset(
            spot,
            options,
            n_steps=5,
            train_end_date=spot.loc[29, "date"],
            validation_end_date=spot.loc[59, "date"],
            embargo_steps=4,
            selection_min_dte=1,
            selection_max_dte=1000,
            atm_target_dte=30,
        )

        adjustment_date = spot.loc[adjustment_step, "date"]
        crossing = metadata[
            (pd.to_datetime(metadata["start_date"]) < adjustment_date)
            & (pd.to_datetime(metadata["end_date"]) >= adjustment_date)
        ]
        self.assertGreater(len(crossing), 0)
        self.assertTrue(crossing["optionid"].eq(200).all())
        self.assertTrue(paths.groupby("episode_index")["strike"].nunique().eq(1).all())
        self.assertTrue(metadata["contract_unique_strikes"].eq(1).all())


if __name__ == "__main__":
    unittest.main()
