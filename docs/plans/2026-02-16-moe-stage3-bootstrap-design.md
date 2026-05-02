# MoE Stage-3 Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bootstrap the codebase from seed-averaged PPO to an MoE-ready pipeline by adding an 8-expert manifest, diverse stage-1 expert training entrypoint, and environment hooks for expert diversity.

**Architecture:** Introduce a manifest-driven stage-1 trainer that supports PPO/A2C/SAC experts with per-expert data slices, feature masks, and reward-shaping profiles. Keep existing ensemble path intact while adding additive modules under `crypto_trader/moe`. Extend `TradingEnv` with optional feature masking and configurable reward component weights.

**Tech Stack:** Python, Stable-Baselines3 (PPO/A2C/SAC), Gymnasium, pytest, YAML config.

---

### Task 1: Add failing tests for MoE config + slicing

**Files:**
- Create: `crypto_trader/tests/test_moe_manifest.py`
- Create: `crypto_trader/tests/test_moe_regime.py`

**Step 1: Write failing tests**
- Test manifest loader returns 8 experts and validates unique IDs.
- Test regime slicer returns non-empty subsets for known slices (`bull`, `bear`, `high_vol`, `low_vol`, `range`).
- Test feature-mask resolver maps named masks to observation indices.

**Step 2: Run test to verify it fails**
Run: `pytest crypto_trader/tests/test_moe_manifest.py crypto_trader/tests/test_moe_regime.py -q`
Expected: FAIL with module/function not found.

**Step 3: Write minimal implementation**
- Add `crypto_trader/moe/manifest.py` and `crypto_trader/moe/regime.py`.
- Add `crypto_trader/configs/moe_experts.yaml`.

**Step 4: Run test to verify it passes**
Run: `pytest crypto_trader/tests/test_moe_manifest.py crypto_trader/tests/test_moe_regime.py -q`
Expected: PASS.

### Task 2: Add failing tests for TradingEnv diversity hooks

**Files:**
- Modify: `crypto_trader/tests/test_execution_safety_local_position.py` (or new dedicated test)
- Create: `crypto_trader/tests/test_trading_env_diversity_hooks.py`

**Step 1: Write failing tests**
- Feature masking zeros out excluded observation slots.
- Reward profile weights affect reward value in a controlled synthetic 2-step market path.

**Step 2: Run test to verify it fails**
Run: `pytest crypto_trader/tests/test_trading_env_diversity_hooks.py -q`
Expected: FAIL because env args do not exist.

**Step 3: Write minimal implementation**
- Update `crypto_trader/envs/trading_env.py` with optional `feature_mask` and `reward_profile` knobs.

**Step 4: Run test to verify it passes**
Run: `pytest crypto_trader/tests/test_trading_env_diversity_hooks.py -q`
Expected: PASS.

### Task 3: Add stage-1 diverse expert trainer

**Files:**
- Create: `crypto_trader/train_moe_stage1.py`
- Modify: `crypto_trader/config.py` (MoE paths + optional defaults)

**Step 1: Write failing tests**
- Ensure trainer can parse manifest and build per-expert training specs without training loops.
- Ensure algorithm mapping resolves to PPO/A2C/SAC.

**Step 2: Run test to verify it fails**
Run: `pytest crypto_trader/tests/test_moe_stage1_spec.py -q`
Expected: FAIL for missing module/functions.

**Step 3: Write minimal implementation**
- Implement data preparation reuse from `train_ensemble.py`.
- Implement `--dry-run` mode that prints resolved expert plan.
- Implement save layout: `checkpoints/moe/stage1/{expert_id}/`.

**Step 4: Run test to verify it passes**
Run: `pytest crypto_trader/tests/test_moe_stage1_spec.py -q`
Expected: PASS.

### Task 4: Verification

**Files:**
- N/A

**Step 1: Run focused tests**
Run: `pytest crypto_trader/tests/test_moe_manifest.py crypto_trader/tests/test_moe_regime.py crypto_trader/tests/test_trading_env_diversity_hooks.py crypto_trader/tests/test_moe_stage1_spec.py -q`
Expected: PASS.

**Step 2: Sanity command**
Run: `python -m crypto_trader.train_moe_stage1 --manifest crypto_trader/configs/moe_experts.yaml --dry-run`
Expected: exits 0 and prints 8 experts resolved.

