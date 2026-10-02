from pathlib import Path
import tempfile
import unittest

from pagb.config import load_config
from pagb.demo import make_demo
from pagb.experiments import evaluate


class SyntheticSmokeTest(unittest.TestCase):
    def test_synthetic_demo_evaluates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_demo(root / "demo", seed=20260928)
            result = evaluate(
                manifest,
                root / "run",
                load_config(Path("configs/smoke.json")),
                data_root=None,
                allow_unverified=True,
                allow_missing_scale=False,
                selected_parameters=None,
                prediction_source="probability",
            )
            self.assertEqual(result["n_total"], 9)
            self.assertEqual(result["n_specimens"], 6)
            self.assertGreaterEqual(result["instance_f1_micro"], 0)
            self.assertLessEqual(result["instance_f1_micro"], 1)


if __name__ == "__main__":
    unittest.main()
