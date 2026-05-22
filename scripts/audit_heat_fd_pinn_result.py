from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_PREDICTION_FIELDS = {"x", "t", "u_pred", "u_ref"}
ERROR_BY_TIME_FIELDS = [
    "time",
    "relative_l2_error",
    "max_abs_error",
    "mean_abs_error",
]
SPECTRAL_FIELDS = [
    "mode",
    "mean_ref_amplitude",
    "mean_pred_amplitude",
    "mean_abs_spectral_error",
    "relative_spectral_error",
    "contribution_to_total_spectral_error",
]


class AuditInputError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit saved heat-equation FD-PINN prediction outputs.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/heat_fd_pinn_m50_s500"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/heat_fd_pinn_m50_s500_audit"),
    )
    return parser.parse_args()


def relative_l2_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference_norm = np.linalg.norm(reference)
    error_norm = np.linalg.norm(reference - prediction)
    if reference_norm == 0.0:
        return float("nan") if error_norm != 0.0 else 0.0
    return float(error_norm / reference_norm)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def read_prediction(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction_path = input_dir / "prediction.npz"
    if not prediction_path.exists():
        raise AuditInputError(f"Missing prediction file: {prediction_path}")

    with np.load(prediction_path) as prediction:
        missing_fields = REQUIRED_PREDICTION_FIELDS.difference(prediction.files)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise AuditInputError(f"prediction.npz is missing required fields: {missing}")

        x = np.array(prediction["x"])
        t = np.array(prediction["t"])
        u_pred = np.array(prediction["u_pred"])
        u_ref = np.array(prediction["u_ref"])

    if u_pred.shape != u_ref.shape:
        raise AuditInputError("u_pred and u_ref must have the same shape")
    if u_pred.shape != (t.size, x.size):
        raise AuditInputError("u_pred and u_ref must have shape (len(t), len(x))")
    return x, t, u_pred, u_ref


def read_metrics(input_dir: Path) -> dict[str, Any]:
    metrics_path = input_dir / "metrics.json"
    if not metrics_path.exists():
        raise AuditInputError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def read_training_history(input_dir: Path) -> list[dict[str, float | int]]:
    history_path = input_dir / "training_history.csv"
    if not history_path.exists():
        raise AuditInputError(f"Missing training history file: {history_path}")

    with history_path.open(newline="", encoding="utf-8") as history_file:
        rows = list(csv.DictReader(history_file))

    required_fields = {"step", "loss"}
    missing_fields = required_fields.difference(rows[0].keys()) if rows else required_fields
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise AuditInputError(f"training_history.csv is missing required fields: {missing}")

    return [{"step": int(row["step"]), "loss": float(row["loss"])} for row in rows]


def make_error_by_time(
    times: np.ndarray,
    u_pred: np.ndarray,
    u_ref: np.ndarray,
) -> list[dict[str, float]]:
    error_rows = []
    abs_error = np.abs(u_ref - u_pred)

    for time_index, time in enumerate(times):
        error_rows.append(
            {
                "time": float(time),
                "relative_l2_error": relative_l2_error(
                    u_ref[time_index],
                    u_pred[time_index],
                ),
                "max_abs_error": float(np.max(abs_error[time_index])),
                "mean_abs_error": float(np.mean(abs_error[time_index])),
            }
        )

    return error_rows


def make_spectral_error_by_mode(
    u_pred: np.ndarray,
    u_ref: np.ndarray,
) -> tuple[list[dict[str, float | int]], np.ndarray, np.ndarray]:
    pred_spectrum = np.fft.rfft(u_pred, axis=1)
    ref_spectrum = np.fft.rfft(u_ref, axis=1)
    spectral_error = np.abs(ref_spectrum - pred_spectrum)
    spectral_error_energy = np.sum(np.square(spectral_error), axis=0)
    total_spectral_error_energy = np.sum(spectral_error_energy)
    rows = []

    for mode in range(ref_spectrum.shape[1]):
        mean_ref_amplitude = float(np.mean(np.abs(ref_spectrum[:, mode])))
        mean_pred_amplitude = float(np.mean(np.abs(pred_spectrum[:, mode])))
        mean_abs_spectral_error = float(np.mean(spectral_error[:, mode]))
        relative_mode_error = safe_ratio(mean_abs_spectral_error, mean_ref_amplitude)
        contribution = spectral_error_energy[mode] / total_spectral_error_energy
        rows.append(
            {
                "mode": mode,
                "mean_ref_amplitude": mean_ref_amplitude,
                "mean_pred_amplitude": mean_pred_amplitude,
                "mean_abs_spectral_error": mean_abs_spectral_error,
                "relative_spectral_error": relative_mode_error,
                "contribution_to_total_spectral_error": float(contribution),
            }
        )

    return rows, pred_spectrum, ref_spectrum


def summarize_frequency_slice(
    pred_spectrum: np.ndarray,
    ref_spectrum: np.ndarray,
    time_index: int,
) -> dict[str, Any]:
    abs_error = np.abs(ref_spectrum[time_index] - pred_spectrum[time_index])
    error_energy = np.square(abs_error)
    total_error_energy = np.sum(error_energy)
    mode_rows = []

    for mode in range(ref_spectrum.shape[1]):
        ref_amplitude = float(np.abs(ref_spectrum[time_index, mode]))
        pred_amplitude = float(np.abs(pred_spectrum[time_index, mode]))
        mode_rows.append(
            {
                "mode": mode,
                "ref_amplitude": ref_amplitude,
                "pred_amplitude": pred_amplitude,
                "abs_spectral_error": float(abs_error[mode]),
                "relative_spectral_error": safe_ratio(
                    float(abs_error[mode]),
                    ref_amplitude,
                ),
                "contribution_to_time_spectral_error": float(
                    error_energy[mode] / total_error_energy,
                ),
            }
        )

    zero_mode = mode_rows[0]
    first_five_contribution = float(np.sum(error_energy[:5]) / total_error_energy)
    high_mode_start = min(50, ref_spectrum.shape[1])
    high_mode_ref_amplitude_ratio = float(
        np.sum(np.abs(ref_spectrum[time_index, high_mode_start:]))
        / np.sum(np.abs(ref_spectrum[time_index]))
    )
    low_mode_residual_error_ratio = float(
        np.sum(error_energy[:5]) / total_error_energy,
    )
    return {
        "zero_mode": zero_mode,
        "first_five_error_contribution": first_five_contribution,
        "high_mode_start": high_mode_start,
        "high_mode_ref_amplitude_ratio": high_mode_ref_amplitude_ratio,
        "low_mode_error_contribution": low_mode_residual_error_ratio,
        "top_error_modes": sorted(
            mode_rows,
            key=lambda row: row["contribution_to_time_spectral_error"],
            reverse=True,
        )[:10],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_diagnostics(
    metrics: dict[str, Any],
    history: list[dict[str, float | int]],
    error_rows: list[dict[str, float]],
    spectral_rows: list[dict[str, float | int]],
    pred_spectrum: np.ndarray,
    ref_spectrum: np.ndarray,
    u_pred: np.ndarray,
    u_ref: np.ndarray,
) -> dict[str, Any]:
    abs_error = np.abs(u_ref - u_pred)
    best_history_row = min(history, key=lambda row: row["loss"])
    max_time_error_index = int(
        np.argmax([row["relative_l2_error"] for row in error_rows])
    )
    top_error_modes = sorted(
        spectral_rows,
        key=lambda row: row["contribution_to_total_spectral_error"],
        reverse=True,
    )[:10]

    return {
        "source_metrics": metrics,
        "total_relative_l2_error": relative_l2_error(u_ref, u_pred),
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
        "relative_l2_error_at_t0": error_rows[0]["relative_l2_error"],
        "relative_l2_error_at_t_final": error_rows[-1]["relative_l2_error"],
        "max_relative_l2_error_over_time": error_rows[max_time_error_index][
            "relative_l2_error"
        ],
        "time_index_of_max_error": max_time_error_index,
        "best_training_step_by_loss": best_history_row["step"],
        "best_training_loss": best_history_row["loss"],
        "final_training_loss": history[-1]["loss"],
        "top_error_modes": top_error_modes,
        "t0_spectral_diagnosis": summarize_frequency_slice(
            pred_spectrum,
            ref_spectrum,
            time_index=0,
        ),
        "t_final_spectral_diagnosis": summarize_frequency_slice(
            pred_spectrum,
            ref_spectrum,
            time_index=-1,
        ),
    }


def run_audit(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    _, times, u_pred, u_ref = read_prediction(input_dir)
    metrics = read_metrics(input_dir)
    history = read_training_history(input_dir)
    error_rows = make_error_by_time(times, u_pred, u_ref)
    spectral_rows, pred_spectrum, ref_spectrum = make_spectral_error_by_mode(
        u_pred,
        u_ref,
    )
    diagnostics = make_diagnostics(
        metrics,
        history,
        error_rows,
        spectral_rows,
        pred_spectrum,
        ref_spectrum,
        u_pred,
        u_ref,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "error_by_time.csv", ERROR_BY_TIME_FIELDS, error_rows)
    write_csv(output_dir / "spectral_error_by_mode.csv", SPECTRAL_FIELDS, spectral_rows)
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as diagnostics_file:
        json.dump(diagnostics, diagnostics_file, indent=2)
        diagnostics_file.write("\n")
    return diagnostics


def main() -> int:
    args = parse_args()

    try:
        diagnostics = run_audit(args.input_dir, args.output_dir)
    except AuditInputError as error:
        print(f"Audit input error: {error}", file=sys.stderr)
        return 1

    print(f"output_dir={args.output_dir}")
    print(f"total_relative_l2_error={diagnostics['total_relative_l2_error']:.6e}")
    print(f"relative_l2_error_at_t0={diagnostics['relative_l2_error_at_t0']:.6e}")
    print(
        "relative_l2_error_at_t_final="
        f"{diagnostics['relative_l2_error_at_t_final']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
