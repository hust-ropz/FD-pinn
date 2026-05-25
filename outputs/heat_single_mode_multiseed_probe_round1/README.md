# Heat Single Mode Multi-Seed Probe Round 1

## 运行摘要

- tested_modes: `[2, 3, 4, 5, 6, 7]`
- tested_seeds: `[0, 1, 2, 3, 4]`
- total_runs: `30`
- steps: `500`
- num_residual: `200`

## Success 判据

- initial_coeff_abs_error < `0.01`
- final_coeff_abs_error < `0.01`
- decay_rate_relative_error < `0.1`

## 每个 mode 的成功率

| Mode | num_success / num_seeds | success_rate | best_seed | median_final_coeff_abs_error | median_decay_rate_relative_error |
| --- | --- | --- | --- | --- | --- |
| `2` | `0 / 5` | `0.0` | `4` | `13.586696790317596` | `1.0000000000000002` |
| `3` | `0 / 5` | `0.0` | `1` | `19.487478372723622` | `1.000000000000001` |
| `4` | `2 / 5` | `0.4` | `3` | `19.11412620376243` | `1.0000000000000007` |
| `5` | `1 / 5` | `0.2` | `3` | `14.651033469773143` | `1.0000000000000002` |
| `6` | `1 / 5` | `0.2` | `2` | `9.520420766491684` | `1.0` |
| `7` | `3 / 5` | `0.6` | `4` | `0.0011715059027437972` | `0.0007399515420396605` |

## 结论

- modes_2_to_5_have_any_successful_seed: `True`
- modes_2_to_5_all_fail_across_seeds: `False`
- modes_6_to_7_remain_stable_across_seeds: `False`
- likely_seed_sensitivity: `True`
- recommended_next_step: 转向 loss/scale 诊断；不建议把 best-of-seeds 当作修复主线。
