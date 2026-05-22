from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.heat_equation_reference import (
    HeatEquationConfig,
    initial_condition,
    periodic_grid,
    relative_l2_error,
    solve_reference,
)

EVAL_TIMES = 101
DTYPE = torch.float64
MODE_DIAGNOSTIC_FIELDS = [
    "step",
    "mode",
    "initial_loss_by_mode",
    "residual_loss_by_mode",
    "weighted_loss_by_mode",
    "ref_t0_abs",
    "pred_t0_abs",
    "ref_tfinal_abs",
    "pred_tfinal_abs",
    "max_pred_abs_over_time",
]
INSTABILITY_EVENT_FIELDS = [
    "step",
    "previous_loss",
    "current_loss",
    "growth_ratio",
    "is_nonfinite",
    "previous_max_mode",
    "previous_max_weighted_loss_by_mode",
    "current_max_mode",
    "current_max_weighted_loss_by_mode",
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
        description="Train the heat-equation FD-PINN mode networks.",
    )
    parser.add_argument("--modes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-residual", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/heat_fd_pinn"),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--initial-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument(
        "--training-schedule",
        choices=["joint", "sequential"],
        default="joint",
    )
    parser.add_argument("--debug-log-modes", action="store_true")
    parser.add_argument("--debug-log-every", type=int, default=10)
    parser.add_argument("--save-best", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, max_modes: int) -> None:
    if args.modes < 1 or args.modes > max_modes:
        raise ValueError(f"modes must be in [1, {max_modes}]")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.num_residual < 1:
        raise ValueError("num_residual must be positive")
    if args.initial_weight < 0.0:
        raise ValueError("initial_weight must be non-negative")
    if args.residual_weight < 0.0:
        raise ValueError("residual_weight must be non-negative")
    if args.debug_log_every < 1:
        raise ValueError("debug_log_every must be positive")


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
    initial_weight: float,
    residual_weight: float,
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
    weighted_loss = initial_weight * initial_loss + residual_weight * residual_loss
    return weighted_loss, initial_loss, residual_loss


def make_mode_networks(modes: int, device: torch.device) -> list[ModeNetwork]:
    return [ModeNetwork().to(device=device, dtype=DTYPE) for _ in range(modes)]


def make_optimizers(networks: list[ModeNetwork]) -> list[torch.optim.LBFGS]:
    return [
        torch.optim.LBFGS(
            network.parameters(),
            max_iter=3,
        )
        for network in networks
    ]


def train_mode_networks(
    networks: list[ModeNetwork],
    initial_targets: torch.Tensor,
    decay_rates: torch.Tensor,
    residual_times: torch.Tensor,
    steps: int,
    initial_weight: float,
    residual_weight: float,
    config: HeatEquationConfig,
    device: torch.device,
    debug_log_modes: bool,
    debug_log_every: int,
    save_best: bool,
) -> tuple[
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    list[dict[str, float | int | bool]],
    dict[str, object] | None,
]:
    optimizers = make_optimizers(networks)
    history = []
    mode_diagnostics = []
    instability_events = []
    best_snapshot = None
    previous_loss = None
    previous_max_mode = None
    previous_max_weighted_loss = None

    for step in range(1, steps + 1):
        step_losses = []
        step_initial_losses = []
        step_residual_losses = []

        for mode_index, (network, optimizer) in enumerate(zip(networks, optimizers)):
            initial_target = initial_targets[mode_index]
            decay_rate = decay_rates[mode_index]

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss, _, _ = mode_losses(
                    network,
                    residual_times,
                    initial_target,
                    decay_rate,
                    initial_weight,
                    residual_weight,
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            loss, initial_loss, residual_loss = mode_losses(
                network,
                residual_times,
                initial_target,
                decay_rate,
                initial_weight,
                residual_weight,
            )
            step_losses.append(loss.detach().item())
            step_initial_losses.append(initial_loss.detach().item())
            step_residual_losses.append(residual_loss.detach().item())

        weighted_loss = float(np.mean(step_losses))
        initial_loss = float(np.mean(step_initial_losses))
        residual_loss = float(np.mean(step_residual_losses))
        history_row = {
            "step": step,
            "training_schedule": "joint",
            "active_mode": -1,
            "loss": weighted_loss,
            "weighted_loss": weighted_loss,
            "initial_loss": initial_loss,
            "residual_loss": residual_loss,
            "initial_weight": initial_weight,
            "residual_weight": residual_weight,
        }
        history.append(history_row)
        current_max_mode = int(np.argmax(step_losses))
        current_max_weighted_loss = step_losses[current_max_mode]

        if previous_loss is not None:
            is_nonfinite = not math.isfinite(weighted_loss)
            growth_ratio = weighted_loss / previous_loss if previous_loss != 0.0 else math.inf
            if is_nonfinite or weighted_loss > 100.0 * previous_loss:
                instability_events.append(
                    {
                        "step": step,
                        "previous_loss": previous_loss,
                        "current_loss": weighted_loss,
                        "growth_ratio": growth_ratio,
                        "is_nonfinite": is_nonfinite,
                        "previous_max_mode": previous_max_mode,
                        "previous_max_weighted_loss_by_mode": previous_max_weighted_loss,
                        "current_max_mode": current_max_mode,
                        "current_max_weighted_loss_by_mode": current_max_weighted_loss,
                    }
                )
        previous_loss = weighted_loss
        previous_max_mode = current_max_mode
        previous_max_weighted_loss = current_max_weighted_loss

        if save_best and (
            best_snapshot is None or weighted_loss < best_snapshot["weighted_loss"]
        ):
            best_snapshot = {
                "step": step,
                "weighted_loss": weighted_loss,
                "initial_loss": initial_loss,
                "residual_loss": residual_loss,
                "state_dicts": copy_mode_state_dicts(networks),
            }

        if debug_log_modes and (step % debug_log_every == 0 or step == steps):
            mode_diagnostics.extend(
                collect_mode_diagnostics(
                    step,
                    networks,
                    initial_targets,
                    decay_rates,
                    step_initial_losses,
                    step_residual_losses,
                    step_losses,
                    config,
                    device,
                )
            )

    return history, mode_diagnostics, instability_events, best_snapshot


def train_sequential_mode_networks(
    networks: list[ModeNetwork],
    initial_targets: torch.Tensor,
    decay_rates: torch.Tensor,
    residual_times: torch.Tensor,
    steps: int,
    initial_weight: float,
    residual_weight: float,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int]],
    list[dict[str, float | int | bool]],
    None,
]:
    optimizers = make_optimizers(networks)
    history = []
    instability_events = []

    for mode_index, (network, optimizer) in enumerate(zip(networks, optimizers)):
        initial_target = initial_targets[mode_index]
        decay_rate = decay_rates[mode_index]
        previous_loss = None

        for step in range(1, steps + 1):
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss, _, _ = mode_losses(
                    network,
                    residual_times,
                    initial_target,
                    decay_rate,
                    initial_weight,
                    residual_weight,
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            loss, initial_loss, residual_loss = mode_losses(
                network,
                residual_times,
                initial_target,
                decay_rate,
                initial_weight,
                residual_weight,
            )
            weighted_loss = loss.detach().item()
            history.append(
                {
                    "step": step,
                    "training_schedule": "sequential",
                    "active_mode": mode_index,
                    "loss": weighted_loss,
                    "weighted_loss": weighted_loss,
                    "initial_loss": initial_loss.detach().item(),
                    "residual_loss": residual_loss.detach().item(),
                    "initial_weight": initial_weight,
                    "residual_weight": residual_weight,
                }
            )

            if previous_loss is not None:
                is_nonfinite = not math.isfinite(weighted_loss)
                growth_ratio = (
                    weighted_loss / previous_loss if previous_loss != 0.0 else math.inf
                )
                if is_nonfinite or weighted_loss > 100.0 * previous_loss:
                    instability_events.append(
                        {
                            "step": step,
                            "previous_loss": previous_loss,
                            "current_loss": weighted_loss,
                            "growth_ratio": growth_ratio,
                            "is_nonfinite": is_nonfinite,
                            "previous_max_mode": mode_index,
                            "previous_max_weighted_loss_by_mode": previous_loss,
                            "current_max_mode": mode_index,
                            "current_max_weighted_loss_by_mode": weighted_loss,
                        }
                    )
            previous_loss = weighted_loss

    return history, [], instability_events, None


def evaluate_global_losses(
    networks: list[ModeNetwork],
    initial_targets: torch.Tensor,
    decay_rates: torch.Tensor,
    residual_times: torch.Tensor,
    initial_weight: float,
    residual_weight: float,
) -> dict[str, float]:
    weighted_losses = []
    initial_losses = []
    residual_losses = []

    for mode_index, network in enumerate(networks):
        loss, initial_loss, residual_loss = mode_losses(
            network,
            residual_times,
            initial_targets[mode_index],
            decay_rates[mode_index],
            initial_weight,
            residual_weight,
        )
        weighted_losses.append(loss.detach().item())
        initial_losses.append(initial_loss.detach().item())
        residual_losses.append(residual_loss.detach().item())

    weighted_loss = float(np.mean(weighted_losses))
    return {
        "loss": weighted_loss,
        "weighted_loss": weighted_loss,
        "initial_loss": float(np.mean(initial_losses)),
        "residual_loss": float(np.mean(residual_losses)),
    }


def reconstruct_prediction(
    networks: list[ModeNetwork],
    config: HeatEquationConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, config.t_max, EVAL_TIMES)
    t_tensor = torch.from_numpy(times[:, None]).to(device=device, dtype=DTYPE)
    spectrum = np.zeros((times.size, config.num_x // 2 + 1), dtype=np.complex128)

    with torch.no_grad():
        for mode_index, network in enumerate(networks):
            mode_prediction = network(t_tensor).cpu().numpy()
            spectrum[:, mode_index] = (
                mode_prediction[:, 0] + 1j * mode_prediction[:, 1]
            )

    u_pred = np.fft.irfft(spectrum, n=config.num_x, axis=1)
    x, u_ref = solve_reference(times, config)
    return x, times, u_pred, u_ref


def copy_mode_state_dicts(networks: list[ModeNetwork]) -> list[dict[str, torch.Tensor]]:
    return [copy.deepcopy(network.state_dict()) for network in networks]


def load_mode_networks(
    state_dicts: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[ModeNetwork]:
    networks = make_mode_networks(len(state_dicts), device)
    for network, state_dict in zip(networks, state_dicts):
        network.load_state_dict(state_dict)
    return networks


def collect_mode_diagnostics(
    step: int,
    networks: list[ModeNetwork],
    initial_targets: torch.Tensor,
    decay_rates: torch.Tensor,
    initial_losses: list[float],
    residual_losses: list[float],
    weighted_losses: list[float],
    config: HeatEquationConfig,
    device: torch.device,
) -> list[dict[str, float | int]]:
    times = np.linspace(0.0, config.t_max, EVAL_TIMES)
    t_tensor = torch.from_numpy(times[:, None]).to(device=device, dtype=DTYPE)
    rows = []

    with torch.no_grad():
        for mode, network in enumerate(networks):
            mode_prediction = network(t_tensor)
            predicted_amplitudes = torch.linalg.vector_norm(mode_prediction, dim=1)
            reference_t0 = torch.linalg.vector_norm(initial_targets[mode])
            reference_tfinal = reference_t0 * torch.exp(
                -decay_rates[mode] * config.t_max
            )
            rows.append(
                {
                    "step": step,
                    "mode": mode,
                    "initial_loss_by_mode": initial_losses[mode],
                    "residual_loss_by_mode": residual_losses[mode],
                    "weighted_loss_by_mode": weighted_losses[mode],
                    "ref_t0_abs": reference_t0.detach().item(),
                    "pred_t0_abs": predicted_amplitudes[0].detach().item(),
                    "ref_tfinal_abs": reference_tfinal.detach().item(),
                    "pred_tfinal_abs": predicted_amplitudes[-1].detach().item(),
                    "max_pred_abs_over_time": torch.max(predicted_amplitudes)
                    .detach()
                    .item(),
                }
            )

    return rows


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_training_history(
    output_dir: Path,
    history: list[dict[str, float | int]],
) -> None:
    with (output_dir / "training_history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "step",
                "training_schedule",
                "active_mode",
                "loss",
                "weighted_loss",
                "initial_loss",
                "residual_loss",
                "initial_weight",
                "residual_weight",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def write_debug_outputs(
    output_dir: Path,
    mode_diagnostics: list[dict[str, float | int]],
    instability_events: list[dict[str, float | int | bool]],
) -> None:
    write_csv(
        output_dir / "mode_training_diagnostics.csv",
        MODE_DIAGNOSTIC_FIELDS,
        mode_diagnostics,
    )
    write_csv(
        output_dir / "instability_events.csv",
        INSTABILITY_EVENT_FIELDS,
        instability_events,
    )


def write_best_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    best_snapshot: dict[str, object],
    config: HeatEquationConfig,
    device: torch.device,
) -> None:
    best_networks = load_mode_networks(best_snapshot["state_dicts"], device)
    x, times, u_pred, u_ref = reconstruct_prediction(best_networks, config, device)
    metrics = {
        "best_step": best_snapshot["step"],
        "best_weighted_loss": best_snapshot["weighted_loss"],
        "best_initial_loss": best_snapshot["initial_loss"],
        "best_residual_loss": best_snapshot["residual_loss"],
        "best_relative_l2_error": relative_l2_error(u_ref, u_pred),
        "initial_weight": args.initial_weight,
        "residual_weight": args.residual_weight,
    }

    np.savez(
        output_dir / "best_prediction.npz",
        x=x,
        t=times,
        u_pred=u_pred,
        u_ref=u_ref,
    )
    with (output_dir / "best_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")


def write_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    history: list[dict[str, float | int]],
    final_losses: dict[str, float | int],
    x: np.ndarray,
    times: np.ndarray,
    u_pred: np.ndarray,
    u_ref: np.ndarray,
    device: torch.device,
) -> dict[str, float | int | str]:
    metrics = {
        "modes": args.modes,
        "steps": args.steps,
        "num_residual": args.num_residual,
        "initial_weight": args.initial_weight,
        "residual_weight": args.residual_weight,
        "training_schedule": args.training_schedule,
        "final_loss": final_losses["loss"],
        "final_initial_loss": final_losses["initial_loss"],
        "final_residual_loss": final_losses["residual_loss"],
        "relative_l2_error": relative_l2_error(u_ref, u_pred),
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": args.seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    np.savez(
        output_dir / "prediction.npz",
        x=x,
        t=times,
        u_pred=u_pred,
        u_ref=u_ref,
    )
    write_training_history(output_dir, history)
    return metrics


def main() -> None:
    args = parse_args()
    config = HeatEquationConfig()
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    spectrum = initial_rfft_spectrum(config)
    k = rfft_wavenumbers(config)
    validate_args(args, max_modes=spectrum.size)

    initial_targets = torch.from_numpy(
        np.column_stack((spectrum[: args.modes].real, spectrum[: args.modes].imag))
    ).to(device=device, dtype=DTYPE)
    decay_rates = torch.from_numpy(((config.a * k[: args.modes]) ** 2)).to(
        device=device,
        dtype=DTYPE,
    )
    residual_times = torch.rand(
        (args.num_residual, 1),
        dtype=DTYPE,
        device=device,
    ) * config.t_max

    networks = make_mode_networks(args.modes, device)
    if args.training_schedule == "joint":
        history, mode_diagnostics, instability_events, best_snapshot = train_mode_networks(
            networks,
            initial_targets,
            decay_rates,
            residual_times,
            args.steps,
            args.initial_weight,
            args.residual_weight,
            config,
            device,
            args.debug_log_modes,
            args.debug_log_every,
            args.save_best,
        )
        final_losses = history[-1]
    else:
        history, mode_diagnostics, instability_events, best_snapshot = (
            train_sequential_mode_networks(
                networks,
                initial_targets,
                decay_rates,
                residual_times,
                args.steps,
                args.initial_weight,
                args.residual_weight,
            )
        )
        final_losses = evaluate_global_losses(
            networks,
            initial_targets,
            decay_rates,
            residual_times,
            args.initial_weight,
            args.residual_weight,
        )
    x, times, u_pred, u_ref = reconstruct_prediction(networks, config, device)
    metrics = write_outputs(
        args,
        args.output_dir,
        history,
        final_losses,
        x,
        times,
        u_pred,
        u_ref,
        device,
    )
    if args.debug_log_modes:
        write_debug_outputs(args.output_dir, mode_diagnostics, instability_events)
    if args.save_best and best_snapshot is not None:
        write_best_outputs(args, args.output_dir, best_snapshot, config, device)

    print(f"output_dir={args.output_dir}")
    print(f"final_loss={metrics['final_loss']:.6e}")
    print(f"relative_l2_error={metrics['relative_l2_error']:.6e}")


if __name__ == "__main__":
    main()
