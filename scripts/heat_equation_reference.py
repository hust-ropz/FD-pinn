from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeatEquationConfig:
    x_min: float = -20.0
    x_max: float = 20.0
    t_max: float = 5.0
    num_x: int = 256
    a: float = 1.0
    paper_wavenumber_threshold: float = 8.0
    paper_kept_nonnegative_modes: int = 50

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        return self.length / self.num_x


def periodic_grid(config: HeatEquationConfig) -> np.ndarray:
    return np.linspace(config.x_min, config.x_max, config.num_x, endpoint=False)


def sech(values: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(values)


def initial_condition(x: np.ndarray) -> np.ndarray:
    return 2.0 * sech(x)


def wavenumbers(config: HeatEquationConfig) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(config.num_x, d=config.dx)


def truncate_nonnegative_modes(spectrum: np.ndarray, keep_modes: int) -> np.ndarray:
    if keep_modes < 1 or keep_modes > spectrum.size // 2:
        raise ValueError("keep_modes must keep at least zero mode and exclude Nyquist mode")

    truncated = np.zeros_like(spectrum)
    truncated[:keep_modes] = spectrum[:keep_modes]
    if keep_modes > 1:
        truncated[-(keep_modes - 1) :] = spectrum[-(keep_modes - 1) :]
    return truncated


def evolve_spectrum(
    initial_spectrum: np.ndarray,
    k: np.ndarray,
    times: np.ndarray,
    a: float,
) -> np.ndarray:
    decay = np.exp(-((a * k[:, None]) ** 2) * times[None, :])
    return initial_spectrum[:, None] * decay


def solve_reference(
    times: np.ndarray,
    config: HeatEquationConfig | None = None,
    keep_modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = config or HeatEquationConfig()
    times = np.asarray(times, dtype=float)
    x = periodic_grid(config)
    spectrum = np.fft.fft(initial_condition(x))

    if keep_modes is not None:
        spectrum = truncate_nonnegative_modes(spectrum, keep_modes)

    evolved = evolve_spectrum(spectrum, wavenumbers(config), times, config.a)
    solution = np.fft.ifft(evolved, axis=0).real.T
    return x, solution


def relative_l2_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(np.linalg.norm(reference - approximation) / np.linalg.norm(reference))


def main() -> None:
    config = HeatEquationConfig()
    times = np.array([0.0, config.t_max])
    x, full_solution = solve_reference(times, config)
    _, truncated_solution = solve_reference(
        times,
        config,
        keep_modes=config.paper_kept_nonnegative_modes,
    )
    k = wavenumbers(config)
    max_kept_k = k[config.paper_kept_nonnegative_modes - 1]

    print("Heat equation spectral reference")
    print(f"x_grid={x.size}, x_range=[{x[0]:.1f}, {config.x_max:.1f})")
    print(f"t_range=[0.0, {config.t_max:.1f}], a={config.a:.1f}")
    print(
        "paper_truncation="
        f"kt={config.paper_wavenumber_threshold:.1f}, "
        f"kept_nonnegative_modes={config.paper_kept_nonnegative_modes}, "
        f"max_kept_k={max_kept_k:.6f}"
    )
    print(
        "relative_l2_error_truncated_vs_full="
        f"t0:{relative_l2_error(full_solution[0], truncated_solution[0]):.6e}, "
        f"tmax:{relative_l2_error(full_solution[-1], truncated_solution[-1]):.6e}"
    )


if __name__ == "__main__":
    main()
