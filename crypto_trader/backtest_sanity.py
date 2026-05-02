"""
回测自检工具 - 实现量化清单第八层三大方法
============================================================
方法 A: 信号延迟冲击测试
  - 将 Signal_Proba 再 shift(1)，观察策略收益是否崩溃
  - 若延迟一天后年化从30%跌到3%，几乎肯定有未来函数

方法 B: 随机信号对照测试
  - 用随机动作替代模型，验证回测框架本身无系统性偏差
  - 若随机信号也能盈利，说明框架有问题（如交易成本过低）

方法 E: 参数敏感性分析
  - 对核心参数做 ±20% 扰动，观察夏普比率波动幅度
  - 剧烈波动 = 过拟合到特定参数

用法:
    cd /Users/cuiqingsong/强化学习_副本
    python -m crypto_trader.backtest_sanity \\
        --manifest crypto_trader/configs/moe_experts.yaml \\
        --data-path crypto_trader/data_moe_20200101_20260216_oos20.csv
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PYPLOT = True
except ImportError:
    HAS_PYPLOT = False

try:
    from crypto_trader.backtest_moe import backtest_moe
    from crypto_trader.config import get_default_config, load_config
    from crypto_trader.risk_manager import RiskManager
    from crypto_trader.envs.trading_env import TradingEnv
    from crypto_trader.asset_profile import get_asset_profile, infer_asset_key
except ImportError:
    from backtest_moe import backtest_moe
    from config import get_default_config, load_config
    from risk_manager import RiskManager
    from envs.trading_env import TradingEnv
    from asset_profile import get_asset_profile, infer_asset_key


# ──────────────────────────────────────────────────────────────
# 辅助：运行一次简单的随机/固定动作回测
# ──────────────────────────────────────────────────────────────

def _run_env_with_action_fn(
    df: pd.DataFrame,
    action_fn,
    config,
    symbol: str,
    enable_kill_switch: bool = True,
    initial_balance: float = 10000.0,
) -> Dict[str, float]:
    """在给定 df 上运行一个动作函数并返回汇总指标。"""
    from stable_baselines3.common.vec_env import DummyVecEnv

    profile = get_asset_profile(symbol)
    cfg = profile.env

    rm = RiskManager(
        max_drawdown_limit=config.risk.max_drawdown_limit,
        freeze_period_steps=config.risk.freeze_period_steps,
        tier1_drawdown=config.risk.tier1_drawdown,
        tier1_limit=config.risk.tier1_limit,
        tier2_drawdown=config.risk.tier2_drawdown,
        tier2_limit=config.risk.tier2_limit,
        survival_drawdown=config.risk.survival_drawdown,
        survival_limit=config.risk.survival_limit,
    )

    env = TradingEnv(
        df,
        initial_balance=initial_balance,
        risk_manager=rm,
        symbol=symbol,
        atr_floor=cfg.atr_floor,
        vol_scale_min=cfg.vol_scale_min,
        vol_scale_max=cfg.vol_scale_max,
        target_atr_pct=cfg.target_atr_pct,
        tau=cfg.tau,
        delta_max=cfg.delta_max,
        cooldown_n=cfg.cooldown_n,
        k_single=cfg.k_single,
        funding_daily=cfg.funding_daily,
        enable_kill_switch=enable_kill_switch,
    )

    obs, _ = env.reset()
    net_worths = [initial_balance]
    done = False
    step = 0
    while not done:
        action = action_fn(obs, step)
        obs, _, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        net_worths.append(float(info["net_worth"]))
        done = terminated or truncated
        step += 1

    nw = np.array(net_worths)
    peak = np.maximum.accumulate(nw)
    max_dd = float(((peak - nw) / peak).max())
    total_ret = float((nw[-1] - initial_balance) / initial_balance)

    # Sharpe estimation (daily)
    daily_rets = np.diff(nw) / nw[:-1]
    sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-9) * np.sqrt(252))

    return {
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "n_steps": step,
        "final_net_worth": float(nw[-1]),
    }


# ──────────────────────────────────────────────────────────────
# 方法 A: 信号延迟冲击测试
# ──────────────────────────────────────────────────────────────

def method_a_signal_delay_test(
    manifest_path: Path,
    data_path: str,
    stage1_root: str,
    stage2_root: str,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    将 Signal_Proba 再 shift(1)（即比原始数据多延迟一天），对比策略收益变化。
    
    诊断规则：
    - 若延迟后 total_return 大幅下降 (>50%)：存在未来函数的强烈信号 ⚠️
    - 若延迟后 total_return 下降但在合理范围内：可能正常的信号衰减
    - 若延迟后 total_return 几乎不变：策略可能过于鲁棒，或不依赖该信号
    """
    print("\n" + "=" * 60)
    print("【方法A】信号延迟冲击测试")
    print("=" * 60)

    # 原始回测
    print("→ 运行原始回测...")
    original = backtest_moe(
        manifest_path=manifest_path,
        stage1_root=stage1_root,
        stage2_root=stage2_root,
        data_path=data_path,
        symbol=symbol,
        config_path=config_path,
    )
    if "error" in original:
        print(f"  ❌ 原始回测失败: {original['error']}")
        return {"status": "error", "detail": original}

    # 延迟数据回测
    delayed_path = data_path.replace(".csv", "_delayed.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if "Signal_Proba" not in df.columns:
        print("  ⚠️ 数据中无 Signal_Proba 列，跳过方法A")
        return {"status": "skipped", "reason": "no Signal_Proba column"}

    df_delayed = df.copy()
    df_delayed["Signal_Proba"] = df_delayed["Signal_Proba"].shift(1).fillna(method="bfill")
    # 注: 此处对 _delayed.csv 允许一次 bfill 仅用于首行填充（不影响策略测试逻辑）
    # 因为 shift(1) 首行变成 NaN，需要一个占位值
    df_delayed.to_csv(delayed_path)

    print("→ 运行延迟版回测 (Signal_Proba 额外 shift 1)...")
    delayed = backtest_moe(
        manifest_path=manifest_path,
        stage1_root=stage1_root,
        stage2_root=stage2_root,
        data_path=delayed_path,
        symbol=symbol,
        config_path=config_path,
    )

    # 清理临时文件
    Path(delayed_path).unlink(missing_ok=True)

    if "error" in delayed:
        print(f"  ❌ 延迟回测失败: {delayed['error']}")
        return {"status": "error", "detail": delayed}

    orig_ret = original["total_return"]
    delay_ret = delayed["total_return"]
    delta = orig_ret - delay_ret
    pct_drop = (delta / (abs(orig_ret) + 1e-9)) * 100

    print(f"\n  原始 total_return: {orig_ret*100:.2f}%")
    print(f"  延迟 total_return: {delay_ret*100:.2f}%")
    print(f"  绝对差值:          {delta*100:.2f}%")
    print(f"  相对下降:          {pct_drop:.1f}%")

    if pct_drop > 50:
        verdict = "⚠️ 高风险: 延迟一天后收益下降>50%，可能存在未来函数！"
    elif pct_drop > 20:
        verdict = "⚡ 中风险: 延迟后收益有明显下降，建议仔细检查信号生成逻辑"
    else:
        verdict = "✅ 低风险: 信号延迟影响在合理范围内"

    print(f"\n  诊断: {verdict}")

    return {
        "status": "ok",
        "original_return": orig_ret,
        "delayed_return": delay_ret,
        "absolute_delta": delta,
        "pct_drop": pct_drop,
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────
# 方法 B: 随机信号对照测试
# ──────────────────────────────────────────────────────────────

def method_b_random_baseline_test(
    data_path: str,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
    n_runs: int = 10,
    seed: int = 42,
) -> Dict:
    """
    用随机动作替换策略，验证回测框架是否存在系统性偏差。
    
    诊断规则：
    - 若随机均值 total_return > 0 且显著（>5%）：框架可能有偏差（如成本过低） ⚠️
    - 若随机结果围绕 0 上下波动：框架基本无偏 ✅
    """
    print("\n" + "=" * 60)
    print("【方法B】随机信号对照测试")
    print("=" * 60)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if "Signal_Proba" not in df.columns:
        # Add dummy column if absent
        df["Signal_Proba"] = 0.5

    config = load_config(config_path) if config_path else get_default_config()
    sym = symbol or infer_asset_key(data_path, interval=config.data.interval)

    np.random.seed(seed)
    random.seed(seed)

    returns = []
    for i in range(n_runs):
        rng = np.random.default_rng(seed + i)

        def random_action(obs, step, _rng=rng):  # noqa: E731
            return float(_rng.uniform(-1.0, 1.0))

        result = _run_env_with_action_fn(df, random_action, config, sym)
        returns.append(result["total_return"])
        print(f"  Run {i+1:2d}: total_return={result['total_return']*100:+.2f}%  "
              f"sharpe={result['sharpe']:+.2f}  max_dd={result['max_drawdown']*100:.1f}%")

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    positive_frac = float(np.mean(np.array(returns) > 0))

    print(f"\n  随机基线: 均值={mean_ret*100:+.2f}%  标准差={std_ret*100:.2f}%  "
          f"正收益概率={positive_frac*100:.0f}%")

    if mean_ret > 0.05:
        verdict = "⚠️ 高风险: 随机信号均值收益>5%，回测框架可能存在系统偏差！检查成本模型"
    elif mean_ret > 0.01:
        verdict = "⚡ 中风险: 随机信号略有正收益，关注成本模型是否充分"
    else:
        verdict = "✅ 低风险: 随机信号围绕零值波动，框架基本无偏"

    print(f"  诊断: {verdict}")

    return {
        "status": "ok",
        "n_runs": n_runs,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "positive_fraction": positive_frac,
        "all_returns": returns,
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────
# 方法 E: 参数敏感性分析
# ──────────────────────────────────────────────────────────────

def method_e_param_sensitivity(
    manifest_path: Path,
    data_path: str,
    stage1_root: str,
    stage2_root: str,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
    perturb_pct: float = 0.20,
) -> Dict:
    """
    对核心风控/成本参数做 ±perturb_pct 扰动，观察 total_return 和 sharpe 波动幅度。
    
    待扰动参数：k_single (手续费), tau (迟滞), delta_max (步进限制)
    
    诊断规则：
    - 夏普比率在扰动下波动 < 20%：参数鲁棒 ✅
    - 夏普比率波动 20-50%：中等敏感性，需关注
    - 夏普比率波动 > 50%：高度过拟合到特定参数 ⚠️
    """
    print("\n" + "=" * 60)
    print(f"【方法E】参数敏感性分析 (±{perturb_pct*100:.0f}%扰动)")
    print("=" * 60)

    # 原始回测
    print("→ 运行基准回测...")
    baseline = backtest_moe(
        manifest_path=manifest_path,
        stage1_root=stage1_root,
        stage2_root=stage2_root,
        data_path=data_path,
        symbol=symbol,
        config_path=config_path,
    )
    if "error" in baseline:
        print(f"  ❌ 基准回测失败: {baseline['error']}")
        return {"status": "error"}

    baseline_ret = baseline["total_return"]
    print(f"  基准 total_return: {baseline_ret*100:.2f}%  max_dd: {baseline['max_dd']*100:.2f}%")

    # 通过 config 修改参数（rebuild config for each perturbation）
    config = load_config(config_path) if config_path else get_default_config()
    sym = symbol or infer_asset_key(data_path, interval=config.data.interval)
    profile = get_asset_profile(sym)
    env_cfg = profile.env

    params_to_test = {
        "k_single (手续费率)":   ("env.k_single", env_cfg.k_single),
        "tau (迟滞阈值)":         ("env.tau", env_cfg.tau),
        "delta_max (步进限制)":   ("env.delta_max", env_cfg.delta_max),
        "funding_daily (资金费)": ("env.funding_daily", env_cfg.funding_daily),
    }

    results = {}
    for label, (_, base_val) in params_to_test.items():
        row = {}
        for direction, mult in [("+", 1 + perturb_pct), ("-", 1 - perturb_pct)]:
            perturbed_val = base_val * mult
            row[direction] = perturbed_val

            # 注意: 此处只能通过 env_kwargs override 传入，不修改 config 文件
            # backtest_moe 目前固定读 asset_profile，需要记录相对变化作为参考
            # TODO: 当 backtest_moe 支持 env_kwargs_override 参数时升级此处
            print(f"  {label} {direction}20% ({base_val:.6f} → {perturbed_val:.6f}): [需 backtest_moe 支持 env_kwargs_override]")

        results[label] = {"base": base_val, "+20%": base_val * (1 + perturb_pct), "-20%": base_val * (1 - perturb_pct)}

    verdict = (
        "⚡ 注意: 完整的参数敏感性测试需要 backtest_moe 支持 env_kwargs_override 参数。\n"
        "   当前版本已列出待测参数和其 ±20% 范围，供手动验证参考。\n"
        "   快速验证方法：临时修改 asset_profile 中对应参数值，重跑回测，比较结果。"
    )

    print(f"\n  待扰动参数 (±{perturb_pct*100:.0f}%):")
    for label, vals in results.items():
        print(f"    {label}: base={vals['base']:.6f}  +20%={vals['+20%']:.6f}  -20%={vals['-20%']:.6f}")
    print(f"\n  {verdict}")

    return {
        "status": "partial",
        "baseline_return": baseline_ret,
        "params": results,
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────
# 汇总报告
# ──────────────────────────────────────────────────────────────

def run_sanity_checks(
    manifest_path: Path,
    data_path: str,
    stage1_root: str,
    stage2_root: str,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
    skip_method_a: bool = False,
    skip_method_b: bool = False,
    skip_method_e: bool = False,
    n_random_runs: int = 10,
) -> Dict:
    """运行所有自检方法并输出汇总报告。"""
    print("\n" + "🔍" * 30)
    print("  量化回测自检报告 (第八层: 系统性自检方法)")
    print("🔍" * 30)

    summary = {}

    if not skip_method_a:
        summary["method_a"] = method_a_signal_delay_test(
            manifest_path=manifest_path,
            data_path=data_path,
            stage1_root=stage1_root,
            stage2_root=stage2_root,
            symbol=symbol,
            config_path=config_path,
        )

    if not skip_method_b:
        summary["method_b"] = method_b_random_baseline_test(
            data_path=data_path,
            symbol=symbol,
            config_path=config_path,
            n_runs=n_random_runs,
        )

    if not skip_method_e:
        summary["method_e"] = method_e_param_sensitivity(
            manifest_path=manifest_path,
            data_path=data_path,
            stage1_root=stage1_root,
            stage2_root=stage2_root,
            symbol=symbol,
            config_path=config_path,
        )

    # 打印最终判断
    print("\n" + "=" * 60)
    print("【最终汇总】")
    print("=" * 60)
    verdicts = []
    for method, result in summary.items():
        verdict = result.get("verdict", "N/A")
        verdicts.append(verdict)
        risk = "⚠️" if "高风险" in verdict or "error" in result.get("status", "") else (
               "⚡" if "中风险" in verdict else "✅")
        print(f"  {method.upper()}: {risk} {verdict[:80]}...")

    return summary


# ──────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="量化回测自检工具 (方法A/B/E)")
    parser.add_argument("--manifest", type=str, default="crypto_trader/configs/moe_experts.yaml")
    parser.add_argument("--stage1-root", type=str, default="checkpoints/moe/stable/experts")
    parser.add_argument("--stage2-root", type=str, default="checkpoints/moe/stable/gate")
    parser.add_argument("--data-path", type=str, default="crypto_trader/data_moe_20200101_20260216_oos20.csv")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n-random-runs", type=int, default=10, help="方法B随机基线运行次数")
    parser.add_argument("--skip-a", action="store_true", help="跳过方法A（信号延迟冲击）")
    parser.add_argument("--skip-b", action="store_true", help="跳过方法B（随机信号对照）")
    parser.add_argument("--skip-e", action="store_true", help="跳过方法E（参数敏感性）")
    parser.add_argument("--method", type=str, choices=["a", "b", "e"], default=None,
                        help="只运行指定方法，不指定则全部运行")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    skip_a = args.skip_a or (args.method and args.method != "a")
    skip_b = args.skip_b or (args.method and args.method != "b")
    skip_e = args.skip_e or (args.method and args.method != "e")

    result = run_sanity_checks(
        manifest_path=Path(args.manifest),
        data_path=args.data_path,
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        symbol=args.symbol,
        config_path=args.config,
        skip_method_a=skip_a,
        skip_method_b=skip_b,
        skip_method_e=skip_e,
        n_random_runs=args.n_random_runs,
    )

    output_path = Path("results/sanity_check_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serializable items
    serializable = {k: {sk: sv for sk, sv in v.items() if isinstance(sv, (str, int, float, list, dict, bool, type(None)))}
                    for k, v in result.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告已保存: {output_path}")


if __name__ == "__main__":
    main()
