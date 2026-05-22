import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class HeatFdPinnTrainingSmokeTest(unittest.TestCase):
    def test_training_script_writes_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "heat_fd_pinn_smoke"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train_heat_fd_pinn",
                    "--modes",
                    "3",
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

            metrics_path = output_dir / "metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertTrue(math.isfinite(metrics["relative_l2_error"]))
            self.assertTrue(math.isfinite(metrics["final_loss"]))

            with np.load(output_dir / "prediction.npz") as prediction:
                self.assertTrue({"x", "t", "u_pred", "u_ref"}.issubset(prediction.files))

            with (output_dir / "training_history.csv").open(
                newline="",
                encoding="utf-8",
            ) as history_file:
                reader = csv.DictReader(history_file)
                self.assertTrue(
                    {
                        "step",
                        "training_schedule",
                        "active_mode",
                        "loss",
                        "weighted_loss",
                        "initial_loss",
                        "residual_loss",
                        "initial_weight",
                        "residual_weight",
                    }.issubset(
                        reader.fieldnames or [],
                    )
                )
                self.assertEqual(len(list(reader)), 2)

    def test_training_script_writes_loss_weights(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "heat_fd_pinn_smoke_icw10"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train_heat_fd_pinn",
                    "--modes",
                    "3",
                    "--steps",
                    "2",
                    "--num-residual",
                    "5",
                    "--initial-weight",
                    "10",
                    "--residual-weight",
                    "1",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["initial_weight"], 10.0)
            self.assertEqual(metrics["residual_weight"], 1.0)
            self.assertTrue(math.isfinite(metrics["final_loss"]))

    def test_training_script_writes_debug_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "heat_fd_pinn_debug_smoke"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train_heat_fd_pinn",
                    "--modes",
                    "3",
                    "--steps",
                    "3",
                    "--num-residual",
                    "5",
                    "--initial-weight",
                    "10",
                    "--residual-weight",
                    "1",
                    "--debug-log-modes",
                    "--debug-log-every",
                    "1",
                    "--save-best",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((output_dir / "mode_training_diagnostics.csv").exists())
            self.assertTrue((output_dir / "instability_events.csv").exists())
            self.assertTrue((output_dir / "best_prediction.npz").exists())
            best_metrics = json.loads(
                (output_dir / "best_metrics.json").read_text(encoding="utf-8")
            )
            self.assertTrue(math.isfinite(best_metrics["best_relative_l2_error"]))

    def test_training_script_writes_sequential_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "heat_fd_pinn_seq_smoke"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train_heat_fd_pinn",
                    "--modes",
                    "3",
                    "--steps",
                    "2",
                    "--num-residual",
                    "5",
                    "--training-schedule",
                    "sequential",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["training_schedule"], "sequential")
            self.assertTrue(math.isfinite(metrics["final_loss"]))
            self.assertTrue(math.isfinite(metrics["relative_l2_error"]))


if __name__ == "__main__":
    unittest.main()
