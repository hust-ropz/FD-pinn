import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class HeatSingleModeProbeSmokeTest(unittest.TestCase):
    def test_probe_writes_single_mode_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "heat_single_mode_probe_smoke"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.probe_heat_single_mode_training",
                    "--modes",
                    "2",
                    "--steps",
                    "2",
                    "--num-residual",
                    "5",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((output_dir / "probe_training_history.csv").exists())
            self.assertTrue((output_dir / "probe_mode_summary.csv").exists())
            self.assertTrue((output_dir / "probe_summary.json").exists())

            with (output_dir / "probe_mode_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as summary_file:
                row = next(csv.DictReader(summary_file))
            self.assertTrue(math.isfinite(float(row["final_loss"])))
            self.assertTrue(
                math.isfinite(float(row["relative_complex_error_over_time"]))
            )

            summary = json.loads(
                (output_dir / "probe_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["modes"], [2])

    def test_probe_writes_multi_seed_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "heat_single_mode_probe_multiseed_smoke"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.probe_heat_single_mode_training",
                    "--modes",
                    "2",
                    "--seeds",
                    "0,1",
                    "--steps",
                    "2",
                    "--num-residual",
                    "5",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((output_dir / "multi_seed_probe_summary.csv").exists())
            summary_path = output_dir / "multi_seed_probe_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("success_rate_by_mode", summary)


if __name__ == "__main__":
    unittest.main()
