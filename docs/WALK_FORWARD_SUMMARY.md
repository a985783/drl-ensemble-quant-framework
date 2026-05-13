# Walk-Forward Summary

Generated source report: `crypto_trader/walk_forward/results/walk_forward_moe/summary/final_report.md`

The generated result directory is excluded from the public Git history. This
document preserves the headline evidence needed to interpret the project.

## Verdict

- Verdict: `FAIL`
- Folds completed: 5
- Folds passed: 3/5
- Average alpha: 0.39%
- Target average alpha: 20%

## Fold Results

| Fold | Train Window | Test Window | Alpha | Return | Benchmark Return | MaxDD | Sharpe | Sortino | Pass |
|------|--------------|-------------|------:|------:|-----------------:|------:|-------:|--------:|------|
| fold_1 | 2020-01-01 to 2021-12-31 | 2022-01-01 to 2022-12-31 | 48.09% | -20.14% | -68.23% | 21.80% | -1.12 | -0.81 | yes |
| fold_2 | 2020-01-01 to 2022-12-31 | 2023-01-01 to 2023-12-31 | -72.41% | 15.54% | 87.94% | 7.23% | 0.78 | 1.25 | no |
| fold_3 | 2020-01-01 to 2023-12-31 | 2024-01-01 to 2024-12-31 | -25.94% | 25.04% | 50.98% | 16.48% | 0.99 | 0.95 | no |
| fold_4 | 2020-01-01 to 2024-12-31 | 2025-01-01 to 2025-12-31 | 7.59% | -10.12% | -17.71% | 21.18% | -0.56 | -0.50 | yes |
| fold_5 | 2020-01-01 to 2025-12-31 | 2026-01-01 to 2026-12-31 | 44.62% | 18.23% | -26.39% | 9.82% | 1.36 | 2.13 | yes |

## Interpretation

The result is useful engineering evidence, but it does not support a public claim
of robust outperformance. The strongest public framing is:

> An execution-aware empirical study of regime-aware MoE deep reinforcement
> learning for crypto futures, with explicit walk-forward failure cases, cost
> assumptions, funding assumptions, and operational safety constraints.
