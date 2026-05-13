# GitHub Release Notes

This repository is prepared as a public research and engineering snapshot for an
ETH/USDT perpetual futures reinforcement-learning trading system.

## What Is Included

- Source code under `crypto_trader/`
- Stable 4-expert MoE configuration under `crypto_trader/configs/moe_experts.yaml`
- Small stable model artifacts under `checkpoints/moe/stable/`
- Research and governance notes under `quant_docs/`
- Walk-forward result summary in `docs/WALK_FORWARD_SUMMARY.md`
- Tests under `crypto_trader/tests/`

## What Is Excluded

The following are intentionally not versioned:

- `.env`, `.env.live`, `.env.demo`, and any local credential files
- Live/runtime state such as `trading_state.json`
- Logs under `logs/`, `runs/`, and `daily_summaries/`
- Local result folders under `results/`
- Generated market-data CSV snapshots under `crypto_trader/data_*.csv`
- Repeated walk-forward training checkpoints under `crypto_trader/walk_forward/checkpoints/`
- Internal planning drafts under `.sisyphus/`

## Rebuild Data

Historical OHLCV data is fetched from OKX through `ccxt`. API credentials are not
required for public historical candles, but network access and exchange
availability are required.

```bash
PYTHONPATH=. python -m crypto_trader.scripts.build_moe_dataset \
  --symbol ETH/USDT:USDT \
  --start 2020-01-01 \
  --end 2026-02-16 \
  --interval 1d \
  --output-prefix crypto_trader/data_moe_20200101_20260216
```

This creates:

- `crypto_trader/data_moe_20200101_20260216_full.csv`
- `crypto_trader/data_moe_20200101_20260216_train80.csv`
- `crypto_trader/data_moe_20200101_20260216_oos20.csv`

## Rebuild Walk-Forward Checkpoints

The public repository excludes repeated fold checkpoints. Regenerate them with:

```bash
PYTHONPATH=. python -m crypto_trader.walk_forward.moe_walk_forward
```

## Live Trading Safety

Do not commit live credentials. Real-money trading requires an explicit local
environment file and `CONFIRM_REAL_MONEY=True`. Public users should start with
demo mode and review `quant_docs/risk_plan.md` plus
`quant_docs/readiness_report_final.md` before any live deployment.

## Evidence Boundary

The walk-forward result is not a profitability proof. The retained summary shows
5 folds completed, 3 folds passed, and an overall `FAIL` verdict under the
project criteria. The defensible framing is an execution-aware empirical study,
not a stable-profitability claim.
