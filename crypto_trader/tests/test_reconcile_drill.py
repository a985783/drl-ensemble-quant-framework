import tempfile
from pathlib import Path
import sys

# Add crypto_trader to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import execution_safety as es


def main():
    tmp_dir = Path(tempfile.mkdtemp())
    es.STATE_FILE = tmp_dir / "trading_state.json"

    state = es.load_state()
    state = es.set_local_position(0.5, state)
    es.save_state(state)

    result = es.reconcile(
        exchange_position=0.0,
        local_position=es.get_local_position(state),
        open_orders=[],
        state=state,
        tolerance=0.01,
    )

    assert result.is_consistent is False
    assert es.is_safe_mode(state) is True


if __name__ == "__main__":
    main()
