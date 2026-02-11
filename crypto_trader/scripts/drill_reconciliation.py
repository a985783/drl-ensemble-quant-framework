#!/usr/bin/env python3
"""Offline reconciliation drill (no exchange calls)."""

import argparse
from pathlib import Path
import tempfile
import sys

# Ensure crypto_trader is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import execution_safety as es


def main():
    parser = argparse.ArgumentParser(description="Offline reconciliation drill")
    parser.add_argument("--local", type=float, default=0.5, help="Local position to simulate")
    parser.add_argument("--exchange", type=float, default=0.0, help="Exchange position to simulate")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Mismatch tolerance")
    args = parser.parse_args()

    tmp_dir = Path(tempfile.mkdtemp())
    es.STATE_FILE = tmp_dir / "trading_state.json"

    state = es.load_state()
    state = es.set_local_position(args.local, state)

    result = es.reconcile(
        exchange_position=args.exchange,
        local_position=es.get_local_position(state),
        open_orders=[],
        state=state,
        tolerance=args.tolerance,
    )

    print("Reconcile OK:", result.is_consistent)
    print("Discrepancies:", result.discrepancies)
    print("SAFE_MODE:", es.is_safe_mode(state))


if __name__ == "__main__":
    main()
