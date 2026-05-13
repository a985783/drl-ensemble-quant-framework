import pandas as pd
import numpy as np

# Load test predictions from Gate Network
# Since we didn't save the raw history, let's write a script to simulate the first 50 steps
# and print the raw Gate weights vs Actions vs Returns

def analyze_math():
    from crypto_trader.backtest_moe import backtest_moe
    from pathlib import Path
    
    result = backtest_moe(
        manifest_path=Path("crypto_trader/configs/moe_experts.yaml"),
        stage1_root="checkpoints/moe/stable/experts",
        stage2_root="checkpoints/moe/stable/gate",
        data_path="crypto_trader/data_moe_20200101_20260216_oos20.csv",
        max_steps=50,
        plot_path="results/dummy.png",
        gate_temperature=0.68,
        enable_kill_switch=False
    )
    
    # We want to export the actual portfolio math per step to see how combining 8 negative experts makes a positive.
    # We will modify a quick local copy of backtest_moe or just read the environment's internal logic.
    
    # Let's write a purely diagnostic logic. I will run a script that intercepts step()
    pass

if __name__ == "__main__":
    analyze_math()
