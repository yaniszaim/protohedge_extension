import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_original_synthetic.py"
SPEC = importlib.util.spec_from_file_location("original_synthetic_reproduction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OriginalSyntheticReproductionTests(unittest.TestCase):
    def test_weighted_lower_tail_mean_uses_fractional_boundary_mass(self):
        values = np.array([-4.0, -1.0, 3.0])
        weights = np.array([0.25, 0.50, 0.25])
        self.assertAlmostEqual(
            MODULE.weighted_lower_tail_mean(values, weights, probability=0.5),
            -2.5,
        )

    def test_paper_protocol_is_cvar50_with_reported_model_sizes(self):
        protocol = MODULE.PAPER_PROTOCOL
        self.assertEqual(protocol["epochs"], 800)
        self.assertEqual(protocol["training_samples"], 10000)
        self.assertEqual(protocol["black_scholes_prototypes"], 100)
        self.assertEqual(protocol["stochastic_volatility_prototypes"], 500)
        self.assertIn("50%", protocol["metric"])

    def test_snapshot_exports_paper_era_code_and_prototypes(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            _, manifest = MODULE.prepare_source_snapshot(repo, temp_dir)
            names = {row["git_path"] for row in manifest["files"]}
            self.assertIn("objectives.py", names)
            self.assertIn(MODULE.PROTOTYPE_PATHS["black_scholes"], names)
            self.assertIn(MODULE.PROTOTYPE_PATHS["stochastic_volatility"], names)
            self.assertIn(MODULE.NOTEBOOK_REFERENCE_PATH, names)
            on_disk = json.loads(
                (Path(temp_dir) / "source_snapshot_manifest.json").read_text()
            )
            self.assertEqual(on_disk["source_commit"], manifest["source_commit"])

    def test_notebook_profile_resolves_to_saved_black_scholes_configuration(self):
        args = Namespace(
            profile="notebook-black-scholes",
            source_commit=None,
            models=None,
            output_dir=None,
        )
        args = MODULE._resolve_profile_args(args)
        self.assertEqual(args.source_commit, MODULE.NOTEBOOK_SOURCE_COMMIT)
        self.assertEqual(args.models, list(MODULE.NOTEBOOK_BLACK_SCHOLES_MODELS))
        self.assertIn("black_scholes_notebook_confirmation", str(args.output_dir))


if __name__ == "__main__":
    unittest.main()
