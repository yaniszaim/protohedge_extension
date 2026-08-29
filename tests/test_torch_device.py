import unittest

import numpy as np
import torch

from deephedging.base_torch import resolve_torch_device
from deephedging.run_train_torch import build_experiment_components, default_config


def _tensor_devices(value):
    if isinstance(value, torch.Tensor):
        return {value.device.type}
    if isinstance(value, dict):
        devices = set()
        for child in value.values():
            devices.update(_tensor_devices(child))
        return devices
    if isinstance(value, (list, tuple)):
        devices = set()
        for child in value:
            devices.update(_tensor_devices(child))
        return devices
    return set()


class TorchDeviceTests(unittest.TestCase):
    def _exercise_device(self, requested_device):
        config = default_config()
        config["world"].update({"samples": 32, "steps": 3, "seed": 17})
        config["model"].update({"agent_type": "feed_forward"})
        config["training"].update({"device": requested_device, "seed": 19})
        components = build_experiment_components(config)
        device = components["device"]

        self.assertEqual(_tensor_devices(components["train_data"]), {device.type})
        self.assertEqual(_tensor_devices(components["val_data"]), {device.type})
        self.assertTrue(all(parameter.device.type == device.type for parameter in components["gym"].parameters()))

        history = components["trainer"].train(
            train_data=components["train_data"],
            val_data=components["val_data"],
            train_weights=np.asarray(components["world"].sample_weights),
            val_weights=np.asarray(components["val_world"].sample_weights),
            n_epochs=1,
            epoch_refresh=1,
            batch_size=16,
        )
        self.assertEqual(len(history["losses"]["training"]), 1)
        self.assertTrue(np.isfinite(history["losses"]["training"][0]))
        self.assertTrue(np.isfinite(history["init_val_loss"]))
        self.assertTrue(np.isfinite(history["best_val_loss"]))
        self.assertEqual(len(history["selection_scores"]["val"]), 1)

    def test_explicit_cpu_training_stays_on_cpu(self):
        self._exercise_device("cpu")

    def test_auto_resolves_to_available_backend(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolve_torch_device("auto").type, expected)

    def test_unavailable_cuda_fails_instead_of_silently_using_cpu(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            resolve_torch_device("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA accelerator not available")
    def test_cuda_training_stays_on_cuda(self):
        self._exercise_device("cuda")


if __name__ == "__main__":
    unittest.main()
