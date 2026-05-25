import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class HeatSingleModeMultiSeedProbeSmokeTest(unittest.TestCase):
    def test_multiseed_probe_writes_required_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "heat_single_mode_multiseed_probe_smoke"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.probe_heat_single_mode_multiseed",
                    "--modes",
                    "2",
                    "--seeds",
                    "0",
                    "1",
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

            per_seed_path = output_dir / "per_seed_results.csv"
            per_mode_path = output_dir / "per_mode_summary.csv"
            summary_path = output_dir / "probe_summary.json"
            self.assertTrue(per_seed_path.exists())
            self.assertTrue(per_mode_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue((output_dir / "report.md").exists())

            with per_seed_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 2)
            self.assertTrue(math.isfinite(float(rows[0]["final_loss"])))
            self.assertIn("success_flag", rows[0])

            with per_mode_path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertIn("success_rate", row)
            self.assertIn("median_final_coeff_abs_error", row)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["tested_modes"], [2])
            self.assertEqual(summary["tested_seeds"], [0, 1])
            self.assertEqual(summary["total_runs"], 2)
            self.assertIn("mode_success_rates", summary)


if __name__ == "__main__":
    unittest.main()
