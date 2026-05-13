"""
execution_safety.py - Execution Safety Layer for Live Trading

Provides:
- Idempotent action key generation
- Order state machine with persistence
- Reconciliation (exchange vs local state)
- SAFE_MODE logic (blocks new positions, allows reduce-only)
- Health checks (API failures, clock drift)

Design Principle:
    SAFE_MODE only blocks opening/adding positions.
    Reduce-only operations (reducing/closing positions) are ALWAYS allowed.
"""
import hashlib
import json
import time
import fcntl
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import alerting (with fallback if not available)
try:
    from crypto_trader.alerting import AlertManager
    alerts = AlertManager()
    HAS_ALERTS = True
except Exception:
    HAS_ALERTS = False
    alerts = None

# Constants
STATE_FILE = Path(__file__).parent.parent / "trading_state.json"
ACTION_BUCKET_SECONDS = 300  # 5-minute buckets for timestamp
MAX_API_FAILURES = 3
MAX_CLOCK_DRIFT_SECONDS = 30


class SafetyState(Enum):
    """Safety state of the trading system."""
    NORMAL = "normal"
    SAFE_MODE = "safe_mode"


class OrderState(Enum):
    """State machine for order lifecycle."""
    IDLE = "idle"
    PLACED = "placed"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    MARKET_FALLBACK = "market_fallback"
    FAILED = "failed"


@dataclass
class ReconciliationResult:
    """Result of reconciliation check."""
    is_consistent: bool
    exchange_position: float
    local_position: float
    discrepancies: List[str] = field(default_factory=list)
    open_orders: List[Dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def summary(self) -> str:
        """Get summary string for logging."""
        if self.is_consistent:
            return "OK"
        return f"MISMATCH: exchange={self.exchange_position:.4f}, local={self.local_position:.4f}"


@dataclass
class HealthStatus:
    """Health status of the system."""
    api_fail_count: int = 0
    last_api_success: Optional[float] = None
    last_clock_check: Optional[float] = None
    clock_drift_seconds: float = 0.0
    
    def is_healthy(self) -> bool:
        return self.api_fail_count < MAX_API_FAILURES


@dataclass
class TradingState:
    """Complete trading state for persistence."""
    # Original fields
    last_flip_timestamp: float = 0.0
    
    # Run tracking
    run_id: Optional[str] = None
    
    # Safety state
    safety_state: str = "normal"
    safe_mode_reason: Optional[str] = None
    safe_mode_entered_at: Optional[float] = None
    
    # Pending actions for idempotency
    pending_actions: Dict[str, Dict] = field(default_factory=dict)
    
    # Current order state
    order_state: Dict = field(default_factory=lambda: {
        "current_order_id": None,
        "state": "idle",
        "action_id": None,
        "created_at": None,
        "target_position": None
    })
    
    # Last reconciliation
    last_reconcile: Dict = field(default_factory=lambda: {
        "timestamp": None,
        "is_consistent": True,
        "exchange_pos": 0.0,
        "local_pos": 0.0,
        "summary": "OK"
    })
    
    # Health tracking
    health: Dict = field(default_factory=lambda: {
        "api_fail_count": 0,
        "last_api_success": None,
        "last_clock_check": None,
        "clock_drift": 0.0
    })

    # Locally tracked position (normalized -1 to 1). None means uninitialized.
    local_position: Optional[float] = None


def generate_action_id(
    symbol: str,
    target_position: float,
    timestamp: float,
    run_id: str
) -> str:
    """
    Generate deterministic action ID for idempotency.
    
    Format: {symbol}_{target_bucket}_{ts_bucket}_{run_id_short}_{hash}
    
    Args:
        symbol: Trading symbol (e.g., "ETH-USDT")
        target_position: Target position (-1 to 1)
        timestamp: Unix timestamp
        run_id: Current run identifier
    
    Returns:
        Deterministic action ID string
    """
    # Bucket target position to 2 decimal places (e.g., 0.456 -> 46)
    target_bucket = int(round(target_position * 100))
    
    # Bucket timestamp to 5-minute intervals
    ts_bucket = int(timestamp // ACTION_BUCKET_SECONDS) * ACTION_BUCKET_SECONDS
    
    # Short run_id (first 8 chars)
    run_id_short = run_id[:8] if run_id else "norunid"
    
    # Create deterministic hash for uniqueness within bucket
    hash_input = f"{symbol}:{target_bucket}:{ts_bucket}:{run_id}"
    hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    
    return f"{symbol}_{target_bucket}_{ts_bucket}_{run_id_short}_{hash_suffix}"


def load_state() -> TradingState:
    """Load trading state from file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
            
            # Convert old format to new format
            state = TradingState(
                last_flip_timestamp=data.get("last_flip_timestamp", 0.0),
                run_id=data.get("run_id"),
                safety_state=data.get("safety_state", "normal"),
                safe_mode_reason=data.get("safe_mode_reason"),
                safe_mode_entered_at=data.get("safe_mode_entered_at"),
                pending_actions=data.get("pending_actions", {}),
                order_state=data.get("order_state", TradingState().order_state),
                last_reconcile=data.get("last_reconcile", TradingState().last_reconcile),
                health=data.get("health", TradingState().health),
                local_position=data.get("local_position", None)
            )
            return state
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Failed to load state, using defaults: {e}")
    
    return TradingState()


def save_state(state: TradingState) -> bool:
    """
    Save trading state to file.
    
    Returns:
        True if save succeeded, False otherwise
    """
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write: write to temp file then rename
        temp_file = STATE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(asdict(state), f, indent=4)
        
        temp_file.rename(STATE_FILE)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save state: {e}")
        return False
    finally:
        # Clean up temp file if it still exists
        if 'temp_file' in locals() and temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass


def is_action_pending(action_id: str, state: TradingState) -> bool:
    """Check if an action is already pending (not yet completed)."""
    if action_id in state.pending_actions:
        action = state.pending_actions[action_id]
        status = action.get("status", "unknown")
        return status in ["pending", "placed", "partial"]
    return False


def is_action_completed(action_id: str, state: TradingState) -> bool:
    """Check if an action has already been completed."""
    if action_id in state.pending_actions:
        action = state.pending_actions[action_id]
        status = action.get("status", "unknown")
        return status in ["filled", "completed"]
    return False


def get_local_position(state: Optional[TradingState] = None, default: float = 0.0) -> float:
    """Get locally tracked position, with a safe default if uninitialized."""
    if state is None:
        state = load_state()
    if state.local_position is None:
        return default
    return float(state.local_position)


def set_local_position(position: float, state: Optional[TradingState] = None) -> TradingState:
    """Set locally tracked position and persist state."""
    if state is None:
        state = load_state()
    state.local_position = float(position)
    save_state(state)
    return state


def register_action(action_id: str, target_position: float, state: TradingState) -> TradingState:
    """Register a new action as pending."""
    state.pending_actions[action_id] = {
        "status": "pending",
        "target_position": target_position,
        "created_at": time.time()
    }
    state.order_state["action_id"] = action_id
    state.order_state["target_position"] = target_position
    state.order_state["state"] = "pending"
    return state


def update_action_status(action_id: str, status: str, state: TradingState, 
                         order_id: Optional[str] = None) -> TradingState:
    """Update action status."""
    if action_id in state.pending_actions:
        state.pending_actions[action_id]["status"] = status
        state.pending_actions[action_id]["updated_at"] = time.time()
    
    state.order_state["state"] = status
    if order_id:
        state.order_state["current_order_id"] = order_id
    
    return state


def complete_action(action_id: str, state: TradingState, 
                    fill_price: float = 0.0) -> TradingState:
    """Mark action as completed."""
    if action_id in state.pending_actions:
        state.pending_actions[action_id]["status"] = "filled"
        state.pending_actions[action_id]["completed_at"] = time.time()
        state.pending_actions[action_id]["fill_price"] = fill_price
    
    state.order_state["state"] = "idle"
    state.order_state["current_order_id"] = None
    state.order_state["action_id"] = None
    
    return state


def enter_safe_mode(reason: str, state: Optional[TradingState] = None) -> TradingState:
    """Enter SAFE_MODE with given reason."""
    if state is None:
        state = load_state()
    
    state.safety_state = SafetyState.SAFE_MODE.value
    state.safe_mode_reason = reason
    state.safe_mode_entered_at = time.time()
    
    save_state(state)
    print(f"[SAFE_MODE] Entered: {reason}")

    # 发送SAFE_MODE进入告警
    if HAS_ALERTS and alerts:
        try:
            alerts.send("ERROR", f"🚨 【SAFE_MODE已激活】原因: {reason} | 系统已进入安全模式，仅允许减仓操作")
        except Exception:
            pass

    return state


def exit_safe_mode(state: Optional[TradingState] = None) -> TradingState:
    """Exit SAFE_MODE (manual intervention required)."""
    if state is None:
        state = load_state()
    
    state.safety_state = SafetyState.NORMAL.value
    state.safe_mode_reason = None
    state.safe_mode_entered_at = None
    
    # Reset health counters
    state.health["api_fail_count"] = 0
    
    save_state(state)
    print("[SAFE_MODE] Exited: returned to NORMAL")

    # 发送SAFE_MODE退出告警
    if HAS_ALERTS and alerts:
        try:
            alerts.send("INFO", "✅ 【SAFE_MODE已退出】系统已恢复正常交易模式")
        except Exception:
            pass

    return state


def is_safe_mode(state: Optional[TradingState] = None) -> bool:
    """Check if system is in SAFE_MODE."""
    if state is None:
        state = load_state()
    return state.safety_state == SafetyState.SAFE_MODE.value


def can_execute_action(
    target_position: float,
    current_position: float,
    state: Optional[TradingState] = None
) -> tuple[bool, str]:
    """
    Check if action can be executed.
    
    In SAFE_MODE:
    - Only reduce-only operations are allowed
    - Reduce = moving position closer to zero
    
    Returns:
        (can_execute, reason)
    """
    if state is None:
        state = load_state()
    
    if state.safety_state == SafetyState.NORMAL.value:
        return True, "normal"
    
    # SAFE_MODE logic
    # Reduce-only: target should be closer to zero than current
    current_abs = abs(current_position)
    target_abs = abs(target_position)
    
    # Allow if:
    # 1. Target is closer to zero (reducing)
    # 2. Target is zero (closing)
    # 3. Same sign and smaller magnitude
    is_reducing = target_abs < current_abs
    is_closing = target_position == 0
    is_same_sign_smaller = (
        (current_position > 0 and target_position >= 0 and target_position < current_position) or
        (current_position < 0 and target_position <= 0 and target_position > current_position)
    )
    
    if is_reducing or is_closing or is_same_sign_smaller:
        return True, f"safe_mode_reduce_only: {state.safe_mode_reason}"
    
    return False, f"blocked_by_safe_mode: {state.safe_mode_reason}"


def record_api_success(state: Optional[TradingState] = None) -> TradingState:
    """Record successful API call."""
    if state is None:
        state = load_state()
    
    state.health["api_fail_count"] = 0
    state.health["last_api_success"] = time.time()
    
    return state


def record_api_failure(state: Optional[TradingState] = None) -> TradingState:
    """Record failed API call. May trigger SAFE_MODE."""
    if state is None:
        state = load_state()
    
    state.health["api_fail_count"] = state.health.get("api_fail_count", 0) + 1

    # 发送API失败告警（连续失败时）
    fail_count = state.health["api_fail_count"]
    if HAS_ALERTS and alerts and fail_count >= 2:
        try:
            alerts.send("WARN", f"⚠️ 【API连续失败】当前连续失败次数: {fail_count}/{MAX_API_FAILURES} | 达到{MAX_API_FAILURES}次将进入SAFE_MODE")
        except Exception:
            pass

    if fail_count >= MAX_API_FAILURES:
        state = enter_safe_mode(
            f"API consecutive failures >= {MAX_API_FAILURES}",
            state
        )

    return state


def check_clock_drift(exchange_timestamp_ms: int, state: Optional[TradingState] = None) -> TradingState:
    """
    Check clock drift between local and exchange time.
    
    Args:
        exchange_timestamp_ms: Exchange server timestamp in milliseconds
    """
    if state is None:
        state = load_state()
    
    local_ts = time.time() * 1000
    drift_ms = abs(local_ts - exchange_timestamp_ms)
    drift_seconds = drift_ms / 1000
    
    state.health["last_clock_check"] = time.time()
    state.health["clock_drift"] = drift_seconds
    
    if drift_seconds > MAX_CLOCK_DRIFT_SECONDS:
        state = enter_safe_mode(
            f"Clock drift too high: {drift_seconds:.1f}s > {MAX_CLOCK_DRIFT_SECONDS}s",
            state
        )
    
    return state


def reconcile(
    exchange_position: float,
    local_position: float,
    open_orders: List[Dict],
    state: Optional[TradingState] = None,
    tolerance: float = 0.01
) -> ReconciliationResult:
    """
    Reconcile exchange state with local state.
    
    Args:
        exchange_position: Position from exchange API
        local_position: Locally tracked position
        open_orders: List of open orders from exchange
        state: Trading state (will be updated if inconsistent)
        tolerance: Acceptable difference threshold
    
    Returns:
        ReconciliationResult
    """
    if state is None:
        state = load_state()
    
    discrepancies = []
    
    # Check position difference
    pos_diff = abs(exchange_position - local_position)
    if pos_diff > tolerance:
        discrepancies.append(
            f"Position mismatch: exchange={exchange_position:.4f}, local={local_position:.4f}"
        )
    
    # Check for unexpected open orders
    if open_orders and state.order_state.get("state") == "idle":
        discrepancies.append(
            f"Unexpected open orders: {len(open_orders)} orders while state is idle"
        )
    
    is_consistent = len(discrepancies) == 0
    
    result = ReconciliationResult(
        is_consistent=is_consistent,
        exchange_position=exchange_position,
        local_position=local_position,
        discrepancies=discrepancies,
        open_orders=open_orders
    )
    
    # Update state
    state.last_reconcile = {
        "timestamp": result.timestamp,
        "is_consistent": is_consistent,
        "exchange_pos": exchange_position,
        "local_pos": local_position,
        "summary": result.summary()
    }
    
    # Enter SAFE_MODE if inconsistent
    if not is_consistent:
        state = enter_safe_mode(
            f"Reconciliation failed: {'; '.join(discrepancies)}",
            state
        )
    elif (
        state.safety_state == SafetyState.SAFE_MODE.value
        and isinstance(state.safe_mode_reason, str)
        and state.safe_mode_reason.startswith("Reconciliation failed:")
        and not open_orders
    ):
        # Auto-recover only from stale reconcile-based SAFE_MODE once
        # exchange/local state is healthy again and there are no live orders.
        state = exit_safe_mode(state)
    
    save_state(state)
    
    return result


def cleanup_old_actions(state: TradingState, max_age_hours: int = 24) -> TradingState:
    """Clean up old completed actions to prevent state bloat."""
    cutoff = time.time() - (max_age_hours * 3600)
    
    to_remove = []
    for action_id, action_data in state.pending_actions.items():
        if action_data.get("status") in ["filled", "completed", "failed", "canceled"]:
            completed_at = action_data.get("completed_at", action_data.get("updated_at", 0))
            if completed_at < cutoff:
                to_remove.append(action_id)
    
    for action_id in to_remove:
        del state.pending_actions[action_id]
    
    return state


# Convenience function for external use
def get_safety_status() -> Dict[str, Any]:
    """Get current safety status summary."""
    state = load_state()
    return {
        "safety_state": state.safety_state,
        "safe_mode_reason": state.safe_mode_reason,
        "safe_mode_entered_at": state.safe_mode_entered_at,
        "api_fail_count": state.health.get("api_fail_count", 0),
        "last_reconcile": state.last_reconcile,
        "pending_actions_count": len(state.pending_actions),
        "order_state": state.order_state.get("state", "unknown")
    }
