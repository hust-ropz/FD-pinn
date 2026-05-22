from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.heat_equation_reference import (
    HeatEquationConfig,
    initial_condition,
    periodic_grid,
)

DTYPE = torch.float64
EVAL_TIMES = 101
DECAY_FIT_MIN_AMPLITUDE = 1.0e-12
DEFAULT_MAIN_OUTPUT = Path("outputs/heat_fd_pinn_m50_s500/prediction.npz")
REQUIRED_PREDICTION_FIELDS = {"x", "t", "u_pred", "u_ref"}
HISTORY_FIELDS = [
    "mode",
    "step",
    "loss",
    "initial_loss",
    "residual_loss",
    "relative_complex_error_over_time",
    "pred_t0_abs",
    "ref_t0_abs",
    "pred_tfinal_abs",
    "ref_tfinal_abs",
]
MODE_SUMMARY_FIELDS = [
    "mode",
    "k",
    "k_squared",
    "final_loss",
    "final_initial_loss",
    "final_residual_loss",
    "relative_complex_error_over_time",
    "initial_complex_abs_error",
    "final_complex_abs_error",
    "fitted_decay_rate",
    "decay_rate_error",
    "valid_decay_fit",
    "final_amplitude_ratio_pred_over_ref",
    "probe_initial_complex_abs_error",
    "probe_final_complex_abs_error",
    "probe_fitted_decay_rate",
    "main_initial_complex_abs_error",
    "main_final_complex_abs_error",
    "main_fitted_decay_rate",
]
MULTI_SEED_SUMMARY_FIELDS = [
    "mode",
    "seed",
    "k",
    "k_squared",
    "final_loss",
    "final_initial_loss",
    "final_residual_loss",
    "relative_complex_error_over_time",
    "initial_complex_abs_error",
    "final_complex_abs_error",
    "fitted_decay_rate",
    "decay_rate_error",
    "final_amplitude_ratio_pred_over_ref",
    "success",
]


class ModeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 2),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.layers(t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train isolated heat-equation FD-PINN single mode probes.",
    )
    parser.add_argument("--modes", default="2,3,4,5,6,7")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-residual", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seeds", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/heat_single_mode_probe_m2_7"),
    )
    return parser.parse_args()


def parse_modes(value: str) -> list[int]:
    modes = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        mode = int(text)
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("modes must include at least one integer mode")
    return modes


def parse_seeds(value: str) -> list[int]:
    if not value:
        return []
    return parse_modes(value)


def validate_args(
    modes: list[int],
    steps: int,
    num_residual: int,
    max_mode: int,
) -> None:
    if any(mode < 0 or mode > max_mode for mode in modes):
        raise ValueError(f"modes must be in [0, {max_mode}]")
    if steps < 1:
        raise ValueError("steps must be positive")
    if num_residual < 1:
        raise ValueError("num_residual must be positive")


def initial_rfft_spectrum(config: HeatEquationConfig) -> np.ndarray:
    x = periodic_grid(config)
    return np.fft.rfft(initial_condition(x))


def rfft_wavenumbers(config: HeatEquationConfig) -> np.ndarray:
    return 2.0 * np.pi * np.fft.rfftfreq(config.num_x, d=config.dx)


def mode_losses(
    network: ModeNetwork,
    residual_times: torch.Tensor,
    initial_target: torch.Tensor,
    decay_rate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = residual_times.detach().clone().requires_grad_(True)
    prediction = network(t)
    derivatives = []

    for component_index in range(prediction.shape[1]):
        derivative = torch.autograd.grad(
            prediction[:, component_index],
            t,
            grad_outputs=torch.ones_like(prediction[:, component_index]),
            create_graph=True,
        )[0]
        derivatives.append(derivative.squeeze(-1))

    derivative_prediction = torch.stack(derivatives, dim=1)
    residual = derivative_prediction + decay_rate * prediction
    residual_loss = torch.mean(residual.square())

    t0 = torch.zeros((1, 1), dtype=residual_times.dtype, device=residual_times.device)
    initial_prediction = network(t0)
    initial_loss = torch.mean((initial_prediction - initial_target.unsqueeze(0)).square())
    return initial_loss + residual_loss, initial_loss, residual_loss


def main_seeded_probe_inputs(
    mode: int,
    seed: int,
    num_residual: int,
    config: HeatEquationConfig,
    device: torch.device,
) -> tuple[ModeNetwork, torch.Tensor]:
    torch.manual_seed(seed)
    residual_times = torch.rand(
        (num_residual, 1),
        dtype=DTYPE,
        device=device,
    ) * config.t_max
    network = None
    for _ in range(mode + 1):
        network = ModeNetwork().to(device=device, dtype=DTYPE)
    if network is None:
        raise RuntimeError("mode network initialization failed")
    return network, residual_times


def relative_complex_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference_norm = np.linalg.norm(reference)
    error_norm = np.linalg.norm(reference - prediction)
    if reference_norm == 0.0:
        return float("nan") if error_norm != 0.0 else 0.0
    return float(error_norm / reference_norm)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def fit_decay_rate(times: np.ndarray, amplitudes: np.ndarray) -> tuple[float | None, bool]:
    valid = amplitudes > DECAY_FIT_MIN_AMPLITUDE
    if np.count_nonzero(valid) < 2:
        return None, False
    slope, _ = np.polyfit(times[valid], np.log(amplitudes[valid]), 1)
    return float(-slope), True


def predict_mode(
    network: ModeNetwork,
    times: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    t_tensor = torch.from_numpy(times[:, None]).to(device=device, dtype=DTYPE)
    with torch.no_grad():
        prediction = network(t_tensor).cpu().numpy()
    return prediction[:, 0] + 1j * prediction[:, 1]


def evaluate_prediction(
    prediction: np.ndarray,
    reference: np.ndarray,
    times: np.ndarray,
    k_squared: float,
) -> dict[str, Any]:
    prediction_amplitude = np.abs(prediction)
    reference_amplitude = np.abs(reference)
    fitted_decay_rate, valid_decay_fit = fit_decay_rate(
        times,
        prediction_amplitude,
    )
    return {
        "relative_complex_error_over_time": relative_complex_error(
            reference,
            prediction,
        ),
        "initial_complex_abs_error": float(np.abs(reference[0] - prediction[0])),
        "final_complex_abs_error": float(np.abs(reference[-1] - prediction[-1])),
        "pred_t0_abs": float(prediction_amplitude[0]),
        "ref_t0_abs": float(reference_amplitude[0]),
        "pred_tfinal_abs": float(prediction_amplitude[-1]),
        "ref_tfinal_abs": float(reference_amplitude[-1]),
        "fitted_decay_rate": fitted_decay_rate,
        "decay_rate_error": (
            None
            if fitted_decay_rate is None
            else float(fitted_decay_rate - k_squared)
        ),
        "valid_decay_fit": valid_decay_fit,
        "final_amplitude_ratio_pred_over_ref": safe_ratio(
            float(prediction_amplitude[-1]),
            float(reference_amplitude[-1]),
        ),
    }


def train_single_mode(
    mode: int,
    initial_value: complex,
    k_value: float,
    seed: int,
    args: argparse.Namespace,
    config: HeatEquationConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    network, residual_times = main_seeded_probe_inputs(
        mode,
        seed,
        args.num_residual,
        config,
        device,
    )
    optimizer = torch.optim.LBFGS(network.parameters(), max_iter=3)
    initial_target = torch.tensor(
        [initial_value.real, initial_value.imag],
        dtype=DTYPE,
        device=device,
    )
    decay_rate = torch.tensor((config.a * k_value) ** 2, dtype=DTYPE, device=device)
    k_squared = decay_rate.detach().item()
    eval_times = np.linspace(0.0, config.t_max, EVAL_TIMES)
    reference = initial_value * np.exp(-k_squared * eval_times)
    history = []

    for step in range(1, args.steps + 1):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss, _, _ = mode_losses(
                network,
                residual_times,
                initial_target,
                decay_rate,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        loss, initial_loss, residual_loss = mode_losses(
            network,
            residual_times,
            initial_target,
            decay_rate,
        )
        prediction = predict_mode(network, eval_times, device)
        evaluation = evaluate_prediction(prediction, reference, eval_times, k_squared)
        history.append(
            {
                "mode": mode,
                "step": step,
                "loss": loss.detach().item(),
                "initial_loss": initial_loss.detach().item(),
                "residual_loss": residual_loss.detach().item(),
                "relative_complex_error_over_time": evaluation[
                    "relative_complex_error_over_time"
                ],
                "pred_t0_abs": evaluation["pred_t0_abs"],
                "ref_t0_abs": evaluation["ref_t0_abs"],
                "pred_tfinal_abs": evaluation["pred_tfinal_abs"],
                "ref_tfinal_abs": evaluation["ref_tfinal_abs"],
            }
        )

    final_evaluation = evaluate_prediction(
        prediction,
        reference,
        eval_times,
        k_squared,
    )
    summary = {
        "mode": mode,
        "k": float(k_value),
        "k_squared": k_squared,
        "final_loss": history[-1]["loss"],
        "final_initial_loss": history[-1]["initial_loss"],
        "final_residual_loss": history[-1]["residual_loss"],
        "relative_complex_error_over_time": final_evaluation[
            "relative_complex_error_over_time"
        ],
        "initial_complex_abs_error": final_evaluation[
            "initial_complex_abs_error"
        ],
        "final_complex_abs_error": final_evaluation["final_complex_abs_error"],
        "fitted_decay_rate": final_evaluation["fitted_decay_rate"],
        "decay_rate_error": final_evaluation["decay_rate_error"],
        "valid_decay_fit": final_evaluation["valid_decay_fit"],
        "final_amplitude_ratio_pred_over_ref": final_evaluation[
            "final_amplitude_ratio_pred_over_ref"
        ],
        "probe_initial_complex_abs_error": final_evaluation[
            "initial_complex_abs_error"
        ],
        "probe_final_complex_abs_error": final_evaluation[
            "final_complex_abs_error"
        ],
        "probe_fitted_decay_rate": final_evaluation["fitted_decay_rate"],
    }
    return history, summary


def load_main_metrics(modes: list[int]) -> tuple[dict[int, dict[str, Any]], str | None]:
    if not DEFAULT_MAIN_OUTPUT.exists():
        return {}, f"Missing original prediction file: {DEFAULT_MAIN_OUTPUT}"

    with np.load(DEFAULT_MAIN_OUTPUT) as prediction:
        missing_fields = REQUIRED_PREDICTION_FIELDS.difference(prediction.files)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            return {}, f"Original prediction is missing required fields: {missing}"
        x = np.array(prediction["x"])
        times = np.array(prediction["t"])
        u_pred = np.array(prediction["u_pred"])
        u_ref = np.array(prediction["u_ref"])

    if u_pred.shape != u_ref.shape or u_pred.shape != (times.size, x.size):
        return {}, "Original prediction arrays have incompatible shapes"

    pred_spectrum = np.fft.rfft(u_pred, axis=1)
    ref_spectrum = np.fft.rfft(u_ref, axis=1)
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=dx)
    if any(mode >= pred_spectrum.shape[1] for mode in modes):
        return {}, "Original prediction does not include every requested mode"

    metrics = {}
    for mode in modes:
        metrics[mode] = evaluate_prediction(
            pred_spectrum[:, mode],
            ref_spectrum[:, mode],
            times,
            float(k[mode] ** 2),
        )
    return metrics, None


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mode_failed(row: dict[str, Any]) -> bool:
    decay_rate_error = row["decay_rate_error"]
    wrong_decay = (
        decay_rate_error is None
        or abs(decay_rate_error) > max(1.0e-2, 0.1 * row["k_squared"])
    )
    return row["relative_complex_error_over_time"] > 5.0e-2 or wrong_decay


def failure_statuses(
    rows: list[dict[str, Any]],
    modes: list[int],
) -> dict[int, bool]:
    by_mode = {row["mode"]: row for row in rows}
    return {mode: mode_failed(by_mode[mode]) for mode in modes if mode in by_mode}


def make_conclusion_flags(
    rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    main_available: bool,
) -> dict[str, bool]:
    probe_low_status = failure_statuses(rows, [2, 3, 4, 5])
    probe_high_status = failure_statuses(rows, [6, 7])
    main_status = failure_statuses(main_rows, [row["mode"] for row in rows])
    low_probe_failed = any(probe_low_status.values())
    high_probe_succeeded = (
        set(probe_high_status) == {6, 7} and not any(probe_high_status.values())
    )
    matching_statuses = (
        main_available
        and set(main_status) == {row["mode"] for row in rows}
        and all(
            main_status[mode] == mode_failed(row)
            for mode, row in ((row["mode"], row) for row in rows)
        )
    )
    main_low_failed = any(
        main_status.get(mode, False) for mode in [2, 3, 4, 5]
    )
    return {
        "modes_2_to_5_fail_in_isolated_probe": low_probe_failed,
        "modes_6_to_7_succeed_in_isolated_probe": high_probe_succeeded,
        "isolated_probe_matches_main_failure_pattern": matching_statuses,
        "likely_main_loop_issue": main_available
        and main_low_failed
        and not low_probe_failed,
        "likely_single_mode_optimization_issue": low_probe_failed
        and (not main_available or matching_statuses),
    }


def compare_with_main(
    rows: list[dict[str, Any]],
    main_metrics: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    main_rows = []
    for row in rows:
        main_evaluation = main_metrics.get(row["mode"])
        if main_evaluation is None:
            row["main_initial_complex_abs_error"] = None
            row["main_final_complex_abs_error"] = None
            row["main_fitted_decay_rate"] = None
            continue
        row["main_initial_complex_abs_error"] = main_evaluation[
            "initial_complex_abs_error"
        ]
        row["main_final_complex_abs_error"] = main_evaluation[
            "final_complex_abs_error"
        ]
        row["main_fitted_decay_rate"] = main_evaluation["fitted_decay_rate"]
        main_rows.append(
            {
                "mode": row["mode"],
                "k_squared": row["k_squared"],
                "relative_complex_error_over_time": main_evaluation[
                    "relative_complex_error_over_time"
                ],
                "decay_rate_error": main_evaluation["decay_rate_error"],
            }
        )
    return main_rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    config = HeatEquationConfig()
    device = torch.device("cpu")
    modes = parse_modes(args.modes)
    spectrum = initial_rfft_spectrum(config)
    k = rfft_wavenumbers(config)
    validate_args(modes, args.steps, args.num_residual, max_mode=spectrum.size - 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    rows = []
    for mode in modes:
        mode_history, mode_summary = train_single_mode(
            mode,
            spectrum[mode],
            k[mode],
            args.seed,
            args,
            config,
            device,
        )
        history.extend(mode_history)
        rows.append(mode_summary)

    main_metrics, main_skip_reason = load_main_metrics(modes)
    main_rows = compare_with_main(rows, main_metrics)
    main_available = main_skip_reason is None
    summary = {
        "modes": modes,
        "steps": args.steps,
        "num_residual": args.num_residual,
        "seed": args.seed,
        "main_output_path": str(DEFAULT_MAIN_OUTPUT),
        "main_output_available": main_available,
        "main_output_skip_reason": main_skip_reason,
        "top_failed_modes": sorted(
            rows,
            key=lambda row: row["relative_complex_error_over_time"],
            reverse=True,
        ),
        "modes_2_to_5_summary": [
            row for row in rows if 2 <= row["mode"] <= 5
        ],
        "modes_6_to_7_summary": [
            row for row in rows if 6 <= row["mode"] <= 7
        ],
        "conclusion_flags": make_conclusion_flags(
            rows,
            main_rows,
            main_available,
        ),
    }

    write_csv(args.output_dir / "probe_training_history.csv", HISTORY_FIELDS, history)
    write_csv(args.output_dir / "probe_mode_summary.csv", MODE_SUMMARY_FIELDS, rows)
    with (args.output_dir / "probe_summary.json").open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")
    return summary


def probe_success(row: dict[str, Any]) -> bool:
    decay_rate_error = row["decay_rate_error"]
    if decay_rate_error is None:
        return False
    relative_decay_error = abs(decay_rate_error) / max(row["k_squared"], 1.0e-12)
    return (
        row["relative_complex_error_over_time"] < 1.0e-2
        and relative_decay_error < 0.1
    )


def rows_by_mode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(row["mode"], []).append(row)
    return grouped_rows


def mode_seed_sensitive(rows: list[dict[str, Any]]) -> bool:
    successes = {row["success"] for row in rows}
    return len(successes) > 1


def group_seed_sensitive(
    grouped_rows: dict[int, list[dict[str, Any]]],
    modes: list[int],
) -> bool:
    return any(
        mode_seed_sensitive(grouped_rows[mode])
        for mode in modes
        if mode in grouped_rows
    )


def make_multi_seed_conclusion_flags(
    rows: list[dict[str, Any]],
    grouped_rows: dict[int, list[dict[str, Any]]],
) -> dict[str, bool]:
    low_rows = [row for row in rows if 2 <= row["mode"] <= 5]
    high_rows = [row for row in rows if 6 <= row["mode"] <= 7]
    low_seed_sensitive = group_seed_sensitive(grouped_rows, [2, 3, 4, 5])
    any_low_success = any(row["success"] for row in low_rows)
    high_modes_complete = {row["mode"] for row in high_rows} == {6, 7}
    return {
        "any_seed_succeeds_for_modes_2_to_5": any_low_success,
        "all_seeds_fail_for_modes_2_to_5": bool(low_rows)
        and not any_low_success,
        "modes_6_to_7_robust_across_seeds": high_modes_complete
        and all(row["success"] for row in high_rows),
        "likely_initialization_sensitivity": low_seed_sensitive,
        "likely_loss_scaling_or_objective_issue": bool(low_rows)
        and not any_low_success,
    }


def run_multi_seed_probe(args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    config = HeatEquationConfig()
    device = torch.device("cpu")
    modes = parse_modes(args.modes)
    spectrum = initial_rfft_spectrum(config)
    k = rfft_wavenumbers(config)
    validate_args(modes, args.steps, args.num_residual, max_mode=spectrum.size - 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        for mode in modes:
            _, mode_summary = train_single_mode(
                mode,
                spectrum[mode],
                k[mode],
                seed,
                args,
                config,
                device,
            )
            row = {
                field: mode_summary.get(field)
                for field in MULTI_SEED_SUMMARY_FIELDS
                if field not in {"seed", "success"}
            }
            row["seed"] = seed
            row["success"] = probe_success(mode_summary)
            rows.append(row)

    grouped_rows = rows_by_mode(rows)
    success_rate_by_mode = {
        mode: float(np.mean([row["success"] for row in mode_rows]))
        for mode, mode_rows in grouped_rows.items()
    }
    best_rows_by_mode = {
        mode: min(
            mode_rows,
            key=lambda row: row["relative_complex_error_over_time"],
        )
        for mode, mode_rows in grouped_rows.items()
    }
    summary = {
        "modes": modes,
        "seeds": seeds,
        "steps": args.steps,
        "num_residual": args.num_residual,
        "success_rate_by_mode": success_rate_by_mode,
        "best_seed_by_mode": {
            mode: row["seed"] for mode, row in best_rows_by_mode.items()
        },
        "best_error_by_mode": {
            mode: row["relative_complex_error_over_time"]
            for mode, row in best_rows_by_mode.items()
        },
        "modes_2_to_5_seed_sensitive": group_seed_sensitive(
            grouped_rows,
            [2, 3, 4, 5],
        ),
        "modes_6_to_7_seed_sensitive": group_seed_sensitive(
            grouped_rows,
            [6, 7],
        ),
        "conclusion_flags": make_multi_seed_conclusion_flags(rows, grouped_rows),
    }

    write_csv(
        args.output_dir / "multi_seed_probe_summary.csv",
        MULTI_SEED_SUMMARY_FIELDS,
        rows,
    )
    with (args.output_dir / "multi_seed_probe_summary.json").open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if seeds:
        summary = run_multi_seed_probe(args, seeds)

        print(f"output_dir={args.output_dir}")
        print(f"modes={','.join(str(mode) for mode in summary['modes'])}")
        print(f"seeds={','.join(str(seed) for seed in summary['seeds'])}")
        return

    summary = run_probe(args)

    print(f"output_dir={args.output_dir}")
    print(f"modes={','.join(str(mode) for mode in summary['modes'])}")
    print(
        "modes_2_to_5_fail_in_isolated_probe="
        f"{summary['conclusion_flags']['modes_2_to_5_fail_in_isolated_probe']}"
    )


if __name__ == "__main__":
    main()
