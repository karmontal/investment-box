"""The engine must stop when the host asks it to.

Why this is not theoretical: in a container the engine is PID 1, and the
kernel installs no default signal dispositions for PID 1. A SIGTERM with no
explicit handler is discarded. Before the handler existed, `docker stop` waited
out the full 30 s grace period and then SIGKILLed the process -- measured, both
times, at 30.5 s -- so the scheduler never shut down and the Telegram queue
never drained. After: 1 s, exit code 0, `dropped=0`.

On a VPS that is every upgrade and every host reboot, not an edge case.
"""

from __future__ import annotations

import asyncio
import signal

import pytest

from investment_box.engine.runner import run_forever
from investment_box.engine.state import EngineState


class _State:
    """Stands in for EngineStateMachine; run_forever only reads `.state`."""

    def __init__(self) -> None:
        self.state = EngineState.RUNNING


class StubRunner:
    """The minimum surface `run_forever` touches."""

    def __init__(self) -> None:
        self.state = _State()
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def kill(self) -> None:
        self.state.state = EngineState.KILLED


@pytest.fixture
def runner() -> StubRunner:
    return StubRunner()


def _handlers_installed() -> set[int]:
    loop = asyncio.get_running_loop()
    registered = getattr(loop, "_signal_handlers", None)
    if registered is None:
        pytest.skip("this event loop does not expose signal handlers")
    return set(registered)


class TestShutdownSignals:
    async def test_sigterm_is_registered_while_running(self, runner: StubRunner) -> None:
        """Without this registration PID 1 silently ignores docker stop."""
        task = asyncio.create_task(run_forever(runner))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)

        try:
            installed = _handlers_installed()
            assert signal.SIGTERM in installed, (
                "SIGTERM has no handler; as PID 1 the engine would ignore docker stop "
                "and be SIGKILLed after the grace period"
            )
            assert signal.SIGINT in installed
        finally:
            runner.kill()
            await asyncio.wait_for(task, timeout=5)

    async def test_sigterm_stops_the_engine_and_calls_stop(
        self, runner: StubRunner
    ) -> None:
        """The real path: deliver the signal and require a clean exit.

        Safe to raise against our own process precisely because the handler is
        installed. If the handler regressed, this would not merely fail -- it
        would terminate the run, which is the loudest possible signal that the
        engine has stopped honouring SIGTERM.
        """
        task = asyncio.create_task(run_forever(runner))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        assert signal.SIGTERM in _handlers_installed()

        asyncio.get_running_loop().call_soon(
            lambda: signal.raise_signal(signal.SIGTERM)
        )
        await asyncio.wait_for(task, timeout=5)

        assert runner.started
        assert runner.stopped, "runner.stop() must run so the scheduler shuts down"

    async def test_handlers_are_removed_afterwards(self, runner: StubRunner) -> None:
        """A leaked handler would fire against a dead loop in the next run."""
        task = asyncio.create_task(run_forever(runner))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        runner.kill()
        await asyncio.wait_for(task, timeout=5)

        assert signal.SIGTERM not in _handlers_installed()

    async def test_a_killed_engine_still_exits_without_a_signal(
        self, runner: StubRunner
    ) -> None:
        """The kill switch path must not depend on a signal arriving."""
        task = asyncio.create_task(run_forever(runner))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        runner.kill()

        await asyncio.wait_for(task, timeout=5)
        assert runner.stopped
