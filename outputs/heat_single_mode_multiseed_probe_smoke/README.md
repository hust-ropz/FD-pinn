# Heat Single Mode Multi-Seed Probe Round 1

## 运行摘要

- tested_modes: `[2]`
- tested_seeds: `[0, 1]`
- total_runs: `2`
- steps: `2`
- num_residual: `5`

## Success 判据

- initial_coeff_abs_error < `0.01`
- final_coeff_abs_error < `0.01`
- decay_rate_relative_error < `0.1`

## 每个 mode 的成功率

| Mode | num_success / num_seeds | success_rate | best_seed | median_final_coeff_abs_error | median_decay_rate_relative_error |
| --- | --- | --- | --- | --- | --- |
| `2` | `0 / 2` | `0.0` | `1` | `12.847820712615011` | `0.9986065293366013` |

## 结论

- modes_2_to_5_have_any_successful_seed: `False`
- modes_2_to_5_all_fail_across_seeds: `True`
- modes_6_to_7_remain_stable_across_seeds: `False`
- likely_seed_sensitivity: `False`
- recommended_next_step: 转向 loss/scale 诊断；不建议把 best-of-seeds 当作修复主线。
