from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from scripts.heat_equation_reference import HeatEquationConfig
from scripts.probe_heat_single_mode_training import (
    initial_rfft_spectrum,
    rfft_wavenumbers,
    train_single_mode,
    validate_args,
)

DEFAULT_MODES = [2, 3, 4, 5, 6, 7]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
INITIAL_COEFF_ABS_ERROR_THRESHOLD = 1.0e-2
FINAL_COEFF_ABS_ERROR_THRESHOLD = 1.0e-2
DECAY_RATE_RELATIVE_ERROR_THRESHOLD = 0.1

PER_SEED_FIELDS = [
    "mode",
    "seed",
    "final_loss",
    "final_initial_loss",
    "final_residual_loss",
    "initial_coeff_abs_error",
    "final_coeff_abs_error",
    "fitted_decay_rate",
    "theoretical_decay_rate",
    "decay_rate_relative_error",
    "success_flag",
]
PER_MODE_FIELDS = [
    "mode",
    "num_seeds",
    "num_success",
    "success_rate",
    "best_seed",
    "best_final_coeff_abs_error",
    "median_final_coeff_abs_error",
    "median_decay_rate_relative_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed isolated heat single-mode probes.",
    )
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-residual", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/heat_single_mode_multiseed_probe_round1"),
    )
    return parser.parse_args()


def parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if values is None:
        return default

    parsed = []
    for value in values:
        for item in value.split(","):
            text = item.strip()
            if not text:
                continue
            number = int(text)
            if number not in parsed:
                parsed.append(number)
    if not parsed:
        raise ValueError("list argument must contain at least one integer")
    return parsed


def relative_decay_error(decay_rate_error: float | None, k_squared: float) -> float:
    if decay_rate_error is None:
        return float("inf")
    return float(abs(decay_rate_error) / max(k_squared, 1.0e-12))


def is_success(
    initial_coeff_abs_error: float,
    final_coeff_abs_error: float,
    decay_rate_relative_error: float,
) -> bool:
    return (
        initial_coeff_abs_error < INITIAL_COEFF_ABS_ERROR_THRESHOLD
        and final_coeff_abs_error < FINAL_COEFF_ABS_ERROR_THRESHOLD
        and decay_rate_relative_error < DECAY_RATE_RELATIVE_ERROR_THRESHOLD
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_for_mode(rows: list[dict[str, Any]], mode: int) -> list[dict[str, Any]]:
    return [row for row in rows if row["mode"] == mode]


def summarize_mode(mode: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    mode_rows = rows_for_mode(rows, mode)
    best_row = min(mode_rows, key=lambda row: row["final_coeff_abs_error"])
    num_success = sum(1 for row in mode_rows if row["success_flag"])
    return {
        "mode": mode,
        "num_seeds": len(mode_rows),
        "num_success": num_success,
        "success_rate": float(num_success / len(mode_rows)),
        "best_seed": best_row["seed"],
        "best_final_coeff_abs_error": best_row["final_coeff_abs_error"],
        "median_final_coeff_abs_error": float(
            median(row["final_coeff_abs_error"] for row in mode_rows)
        ),
        "median_decay_rate_relative_error": float(
            median(row["decay_rate_relative_error"] for row in mode_rows)
        ),
    }


def group_has_mixed_success(rows: list[dict[str, Any]], modes: list[int]) -> bool:
    for mode in modes:
        mode_rows = rows_for_mode(rows, mode)
        statuses = {row["success_flag"] for row in mode_rows}
        if len(statuses) > 1:
            return True
    return False


def all_modes_all_success(rows: list[dict[str, Any]], modes: list[int]) -> bool:
    for mode in modes:
        mode_rows = rows_for_mode(rows, mode)
        if not mode_rows or not all(row["success_flag"] for row in mode_rows):
            return False
    return True


def make_summary(
    modes: list[int],
    seeds: list[int],
    steps: int,
    num_residual: int,
    per_seed_rows: list[dict[str, Any]],
    per_mode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    low_rows = [
        row for row in per_seed_rows if row["mode"] in {2, 3, 4, 5}
    ]
    low_any_success = any(row["success_flag"] for row in low_rows)
    low_all_fail = bool(low_rows) and not low_any_success
    high_stable = all_modes_all_success(per_seed_rows, [6, 7])
    seed_sensitive = group_has_mixed_success(per_seed_rows, [2, 3, 4, 5])
    recommended_next_step = (
        "转向 loss/scale 诊断；不建议把 best-of-seeds 当作修复主线。"
        if seed_sensitive or low_all_fail or not high_stable
        else "可以考虑扩大 seed 统计，但仍应先解释 loss/scale 稳定性。"
    )
    return {
        "tested_modes": modes,
        "tested_seeds": seeds,
        "total_runs": len(per_seed_rows),
        "steps": steps,
        "num_residual": num_residual,
        "success_thresholds": {
            "initial_coeff_abs_error": INITIAL_COEFF_ABS_ERROR_THRESHOLD,
            "final_coeff_abs_error": FINAL_COEFF_ABS_ERROR_THRESHOLD,
            "decay_rate_relative_error": DECAY_RATE_RELATIVE_ERROR_THRESHOLD,
        },
        "mode_success_rates": {
            row["mode"]: row["success_rate"] for row in per_mode_rows
        },
        "modes_2_to_5_have_any_successful_seed": low_any_success,
        "modes_2_to_5_all_fail_across_seeds": low_all_fail,
        "modes_6_to_7_remain_stable_across_seeds": high_stable,
        "likely_seed_sensitivity": seed_sensitive,
        "recommended_next_step": recommended_next_step,
    }


def write_output_report(
    output_dir: Path,
    summary: dict[str, Any],
    per_mode_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Heat Single Mode Multi-Seed Probe Round 1",
        "",
        "## 运行摘要",
        "",
        f"- tested_modes: `{summary['tested_modes']}`",
        f"- tested_seeds: `{summary['tested_seeds']}`",
        f"- total_runs: `{summary['total_runs']}`",
        f"- steps: `{summary['steps']}`",
        f"- num_residual: `{summary['num_residual']}`",
        "",
        "## Success 判据",
        "",
        f"- initial_coeff_abs_error < `{INITIAL_COEFF_ABS_ERROR_THRESHOLD}`",
        f"- final_coeff_abs_error < `{FINAL_COEFF_ABS_ERROR_THRESHOLD}`",
        f"- decay_rate_relative_error < `{DECAY_RATE_RELATIVE_ERROR_THRESHOLD}`",
        "",
        "## 每个 mode 的成功率",
        "",
        "| Mode | num_success / num_seeds | success_rate | best_seed | median_final_coeff_abs_error | median_decay_rate_relative_error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in per_mode_rows:
        lines.append(
            "| "
            f"`{row['mode']}` | "
            f"`{row['num_success']} / {row['num_seeds']}` | "
            f"`{row['success_rate']}` | "
            f"`{row['best_seed']}` | "
            f"`{row['median_final_coeff_abs_error']}` | "
            f"`{row['median_decay_rate_relative_error']}` |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- modes_2_to_5_have_any_successful_seed: `{summary['modes_2_to_5_have_any_successful_seed']}`",
            f"- modes_2_to_5_all_fail_across_seeds: `{summary['modes_2_to_5_all_fail_across_seeds']}`",
            f"- modes_6_to_7_remain_stable_across_seeds: `{summary['modes_6_to_7_remain_stable_across_seeds']}`",
            f"- likely_seed_sensitivity: `{summary['likely_seed_sensitivity']}`",
            f"- recommended_next_step: {summary['recommended_next_step']}",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_probe(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    modes = parse_int_list(args.modes, DEFAULT_MODES)
    seeds = parse_int_list(args.seeds, DEFAULT_SEEDS)
    config = HeatEquationConfig()
    spectrum = initial_rfft_spectrum(config)
    k = rfft_wavenumbers(config)
    validate_args(modes, args.steps, args.num_residual, max_mode=spectrum.size - 1)
    device = torch.device("cpu")
    train_args = SimpleNamespace(
        steps=args.steps,
        num_residual=args.num_residual,
    )

    per_seed_rows = []
    for mode in modes:
        for seed in seeds:
            _, mode_summary = train_single_mode(
                mode,
                spectrum[mode],
                k[mode],
                seed,
                train_args,
                config,
                device,
            )
            initial_error = mode_summary["initial_complex_abs_error"]
            final_error = mode_summary["final_complex_abs_error"]
            theoretical_decay = mode_summary["k_squared"]
            decay_rate_relative_error = relative_decay_error(
                mode_summary["decay_rate_error"],
                theoretical_decay,
            )
            success_flag = is_success(
                initial_error,
                final_error,
                decay_rate_relative_error,
            )
            per_seed_rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "final_loss": mode_summary["final_loss"],
                    "final_initial_loss": mode_summary["final_initial_loss"],
                    "final_residual_loss": mode_summary["final_residual_loss"],
                    "initial_coeff_abs_error": initial_error,
                    "final_coeff_abs_error": final_error,
                    "fitted_decay_rate": mode_summary["fitted_decay_rate"],
                    "theoretical_decay_rate": theoretical_decay,
                    "decay_rate_relative_error": decay_rate_relative_error,
                    "success_flag": success_flag,
                }
            )

    per_mode_rows = [summarize_mode(mode, per_seed_rows) for mode in modes]
    summary = make_summary(
        modes,
        seeds,
        args.steps,
        args.num_residual,
        per_seed_rows,
        per_mode_rows,
    )
    return per_seed_rows, per_mode_rows, summary


def main() -> None:
    args = parse_args()
    per_seed_rows, per_mode_rows, summary = run_probe(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "per_seed_results.csv", PER_SEED_FIELDS, per_seed_rows)
    write_csv(args.output_dir / "per_mode_summary.csv", PER_MODE_FIELDS, per_mode_rows)
    with (args.output_dir / "probe_summary.json").open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")
    write_output_report(args.output_dir, summary, per_mode_rows)

    print(f"output_dir={args.output_dir}")
    print(f"total_runs={summary['total_runs']}")
    print(f"likely_seed_sensitivity={summary['likely_seed_sensitivity']}")


if __name__ == "__main__":
    main()
