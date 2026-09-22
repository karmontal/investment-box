"""The scheduled trading engine.

``loop`` runs one cycle, ``scheduler`` decides when, ``state`` tracks whether
the engine may act, and ``kill_switch`` stops everything.
"""

from investment_box.engine.kill_switch import KillResult, KillSwitch
from investment_box.engine.live_guard import (
    LIVE_CONFIRMATION_PHRASE,
    ActivationResult,
    LiveActivation,
    LiveGuard,
    LiveTradingRefusedError,
)
from investment_box.engine.loop import CycleResult, TradingCycle
from investment_box.engine.preflight import (
    CheckResult,
    CheckStatus,
    PreflightChecklist,
    PreflightReport,
)
from investment_box.engine.runner import EngineRunner, build_engine, run_forever
from investment_box.engine.scheduler import EngineScheduler, ScheduledJob
from investment_box.engine.state import EngineState, EngineStateMachine, EngineStatus

__all__ = [
    "LIVE_CONFIRMATION_PHRASE",
    "ActivationResult",
    "CheckResult",
    "CheckStatus",
    "CycleResult",
    "EngineRunner",
    "EngineScheduler",
    "EngineState",
    "EngineStateMachine",
    "EngineStatus",
    "KillResult",
    "KillSwitch",
    "LiveActivation",
    "LiveGuard",
    "LiveTradingRefusedError",
    "PreflightChecklist",
    "PreflightReport",
    "ScheduledJob",
    "TradingCycle",
    "build_engine",
    "run_forever",
]
