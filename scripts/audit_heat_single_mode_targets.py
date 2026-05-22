from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_PREDICTION_FIELDS = {"x", "t", "u_pred", "u_ref"}
DIAGNOSTIC_FIELDS = [
    "mode",
    "k",
    "k_squared",
    "ref_initial_real",
    "ref_initial_imag",
    "pred_initial_real",
    "pred_initial_imag",
    "initial_complex_abs_error",
    "final_complex_abs_error",
    "mean_complex_abs_error_over_time",
    "relative_complex_error_over_time",
    "ref_final_amplitude",
    "pred_final_amplitude",
    "final_amplitude_ratio_pred_over_ref",
    "exact_vs_reference_relative_error_by_mode",
    "fitted_decay_rate",
    "decay_rate_error",
    "valid_decay_fit",
]
DECAY_FIT_MIN_AMPLITUDE = 1.0e-12


class AuditInputError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit heat-equation FD-PINN single mode closed-form targets.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/heat_fd_pinn_m50_s500"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/heat_fd_pinn_single_mode_target_audit"),
    )
    parser.add_argument("--modes", type=int, default=50)
    return parser.parse_args()


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


def relative_complex_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference_norm = np.linalg.norm(reference)
    error_norm = np.linalg.norm(reference - prediction)
    if reference_norm == 0.0:
        return float("nan") if error_norm != 0.0 else 0.0
    return float(error_norm / reference_norm)


def fit_decay_rate(times: np.ndarray, amplitudes: np.ndarray) -> tuple[float | None, bool]:
    valid = amplitudes > DECAY_FIT_MIN_AMPLITUDE
    if np.count_nonzero(valid) < 2:
        return None, False
    slope, _ = np.polyfit(times[valid], np.log(amplitudes[valid]), 1)
    return float(-slope), True


def make_closed_form_spectrum(
    ref_spectrum: np.ndarray,
    times: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    return ref_spectrum[0][None, :] * np.exp(-(k[None, :] ** 2) * times[:, None])


def make_mode_rows(
    pred_spectrum: np.ndarray,
    ref_spectrum: np.ndarray,
    closed_form_spectrum: np.ndarray,
    times: np.ndarray,
    k: np.ndarray,
    modes: int,
) -> list[dict[str, Any]]:
    rows = []

    for mode in range(modes):
        prediction = pred_spectrum[:, mode]
        reference = ref_spectrum[:, mode]
        exact = closed_form_spectrum[:, mode]
        prediction_amplitude = np.abs(prediction)
        reference_amplitude = np.abs(reference)
        fitted_decay_rate, valid_decay_fit = fit_decay_rate(
            times,
            prediction_amplitude,
        )
        k_squared = float(k[mode] ** 2)
        rows.append(
            {
                "mode": mode,
                "k": float(k[mode]),
                "k_squared": k_squared,
                "ref_initial_real": float(reference[0].real),
                "ref_initial_imag": float(reference[0].imag),
                "pred_initial_real": float(prediction[0].real),
                "pred_initial_imag": float(prediction[0].imag),
                "initial_complex_abs_error": float(
                    np.abs(reference[0] - prediction[0])
                ),
                "final_complex_abs_error": float(
                    np.abs(reference[-1] - prediction[-1])
                ),
                "mean_complex_abs_error_over_time": float(
                    np.mean(np.abs(reference - prediction))
                ),
                "relative_complex_error_over_time": relative_complex_error(
                    reference,
                    prediction,
                ),
                "ref_final_amplitude": float(reference_amplitude[-1]),
                "pred_final_amplitude": float(prediction_amplitude[-1]),
                "final_amplitude_ratio_pred_over_ref": safe_ratio(
                    float(prediction_amplitude[-1]),
                    float(reference_amplitude[-1]),
                ),
                "exact_vs_reference_relative_error_by_mode": relative_complex_error(
                    exact,
                    reference,
                ),
                "fitted_decay_rate": fitted_decay_rate,
                "decay_rate_error": (
                    None
                    if fitted_decay_rate is None
                    else float(fitted_decay_rate - k_squared)
                ),
                "valid_decay_fit": valid_decay_fit,
            }
        )

    return rows


def top_modes(
    rows: list[dict[str, Any]],
    metric: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row[metric], reverse=True)[:limit]


def make_conclusion_flags(
    exact_vs_reference_total_error: float,
    rows: list[dict[str, Any]],
) -> dict[str, bool]:
    low_mode_rows = [row for row in rows if 2 <= row["mode"] <= 5]
    return {
        "reference_matches_closed_form": exact_vs_reference_total_error < 1.0e-10,
        "low_modes_have_initial_mismatch": any(
            row["initial_complex_abs_error"] > 1.0e-2 for row in low_mode_rows
        ),
        "low_modes_have_wrong_decay_rate": any(
            row["valid_decay_fit"]
            and abs(row["decay_rate_error"]) > max(1.0e-2, 0.1 * row["k_squared"])
            for row in low_mode_rows
        ),
        "low_modes_have_final_amplitude_error": any(
            row["final_complex_abs_error"] > 1.0e-2 for row in low_mode_rows
        ),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(input_dir: Path, output_dir: Path, modes: int) -> dict[str, Any]:
    x, times, u_pred, u_ref = read_prediction(input_dir)
    pred_spectrum = np.fft.rfft(u_pred, axis=1)
    ref_spectrum = np.fft.rfft(u_ref, axis=1)
    if modes < 1 or modes > ref_spectrum.shape[1]:
        raise AuditInputError(f"modes must be in [1, {ref_spectrum.shape[1]}]")

    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=dx)
    closed_form_spectrum = make_closed_form_spectrum(ref_spectrum, times, k)
    rows = make_mode_rows(
        pred_spectrum,
        ref_spectrum,
        closed_form_spectrum,
        times,
        k,
        modes,
    )
    exact_vs_reference_total_error = relative_complex_error(
        closed_form_spectrum[:, :modes],
        ref_spectrum[:, :modes],
    )
    summary = {
        "input_dir": str(input_dir),
        "modes": modes,
        "exact_vs_reference_total_relative_error": exact_vs_reference_total_error,
        "top_modes_by_mean_complex_error": top_modes(
            rows,
            "mean_complex_abs_error_over_time",
        ),
        "top_modes_by_final_complex_error": top_modes(
            rows,
            "final_complex_abs_error",
        ),
        "modes_2_to_5_summary": [
            row for row in rows if 2 <= row["mode"] <= 5
        ],
        "conclusion_flags": make_conclusion_flags(
            exact_vs_reference_total_error,
            rows,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "single_mode_target_diagnostics.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")
    return summary


def main() -> int:
    args = parse_args()

    try:
        summary = run_audit(args.input_dir, args.output_dir, args.modes)
    except AuditInputError as error:
        print(f"Audit input error: {error}", file=sys.stderr)
        return 1

    print(f"output_dir={args.output_dir}")
    print(
        "exact_vs_reference_total_relative_error="
        f"{summary['exact_vs_reference_total_relative_error']:.6e}"
    )
    print(f"reference_matches_closed_form={summary['conclusion_flags']['reference_matches_closed_form']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
