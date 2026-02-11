import tempfile
from pathlib import Path
import sys

# Ensure crypto_trader package is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import execution_safety as es


def main():
    tmp_dir = Path(tempfile.mkdtemp())
    es.STATE_FILE = tmp_dir / "trading_state.json"

    state = es.load_state()
    # local_position should exist and default to 0.0 via getter
    assert hasattr(state, "local_position"), "TradingState missing local_position"
    assert es.get_local_position(state) == 0.0, "Default local_position should be 0.0"

    # Update and persist local_position
    state = es.set_local_position(0.42, state)
    es.save_state(state)

    reloaded = es.load_state()
    assert abs(es.get_local_position(reloaded) - 0.42) < 1e-9, "local_position not persisted"

    # Reconcile should detect mismatch when local != exchange
    result = es.reconcile(
        exchange_position=0.0,
        local_position=es.get_local_position(reloaded),
        open_orders=[],
        state=reloaded,
        tolerance=0.01,
    )
    assert result.is_consistent is False, "Reconcile should flag mismatch"


if __name__ == "__main__":
    main()
