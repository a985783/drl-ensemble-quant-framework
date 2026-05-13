import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add crypto_trader to path if needed (run from project root)
from crypto_trader.backtest_moe import backtest_moe

def main():
    print("Running MoE backtest to inspect gate weights and actions...")
    
    result = backtest_moe(
        manifest_path=Path("crypto_trader/configs/moe_experts.yaml"),
        stage1_root="checkpoints/moe/stable/experts",
        stage2_root="checkpoints/moe/stable/gate",
        data_path="crypto_trader/data_moe_20200101_20260216_oos20.csv",
        plot_path="results/moe_investigate_dump.png",
        gate_temperature=0.68,
        enable_kill_switch=False
    )
    
    if "error" in result:
        print(f"Error running MoE: {result['error']}")
        return
        
    print(f"MoE Total Return: {result['total_return']*100:.2f}%")
    print(f"MoE Max Drawdown: {result['max_dd']*100:.2f}%")
    
    print("\nGate Usage:")
    for eid, usage in result["gate_usage"].items():
        print(f"  {eid}: {usage*100:.2f}%")
        
    print("\nExpert Contributions:")
    for eid, contrib in result["expert_contribution"].items():
        print(f"  {eid}: {contrib:.6f}")

if __name__ == "__main__":
    main()
