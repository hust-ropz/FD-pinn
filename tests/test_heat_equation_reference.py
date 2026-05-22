import unittest

import numpy as np

from scripts.heat_equation_reference import (
    HeatEquationConfig,
    initial_condition,
    periodic_grid,
    solve_reference,
    truncate_nonnegative_modes,
)


class HeatEquationReferenceTests(unittest.TestCase):
    def test_periodic_grid_matches_paper_resolution(self) -> None:
        config = HeatEquationConfig()
        x = periodic_grid(config)

        self.assertEqual(x.shape, (config.num_x,))
        self.assertAlmostEqual(x[0], config.x_min)
        self.assertLess(x[-1], config.x_max)
        self.assertAlmostEqual(x[1] - x[0], config.dx)

    def test_full_reference_recovers_initial_condition(self) -> None:
        x, solution = solve_reference(np.array([0.0]))

        np.testing.assert_allclose(solution[0], initial_condition(x), atol=1e-12)

    def test_truncation_preserves_real_initial_condition(self) -> None:
        config = HeatEquationConfig()
        x = periodic_grid(config)
        spectrum = np.fft.fft(initial_condition(x))
        truncated = truncate_nonnegative_modes(
            spectrum,
            config.paper_kept_nonnegative_modes,
        )

        truncated_initial = np.fft.ifft(truncated)
        self.assertLess(np.max(np.abs(truncated_initial.imag)), 1e-12)


if __name__ == "__main__":
    unittest.main()
