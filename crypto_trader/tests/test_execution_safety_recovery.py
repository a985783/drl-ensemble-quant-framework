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
    state = es.set_local_position(0.42, state)
    es.save_state(state)

    mismatch = es.reconcile(
        exchange_position=0.0,
        local_position=es.get_local_position(state),
        open_orders=[],
        state=state,
        tolerance=0.01,
    )

    assert mismatch.is_consistent is False
    assert es.is_safe_mode(state) is True

    recovered = es.reconcile(
        exchange_position=0.42,
        local_position=0.42,
        open_orders=[],
        state=state,
        tolerance=0.01,
    )

    assert recovered.is_consistent is True
    assert es.is_safe_mode(state) is False, "SAFE_MODE should exit after healthy reconcile"


if __name__ == "__main__":
    main()
