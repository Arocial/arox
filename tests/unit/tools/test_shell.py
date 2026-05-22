import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from arox.plugins.shell import POLL_BASE_DELAY, ShellPlugin


class CaptureIO:
    def __init__(self):
        self.messages: list[str] = []

    async def send(self, msg):
        self.messages.append(msg)

    @contextlib.asynccontextmanager
    async def text_stream(self):
        async def write(delta: str) -> None:
            if delta:
                self.messages.append(delta)

        yield write


class MockAgent:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.agent_io = CaptureIO()
        self._slots: dict = {}

    def provide_slot(self, slot, provider):
        self._slots.setdefault(slot, []).append(provider)

    def get_slot(self, slot):
        return self._slots.get(slot, [])


@pytest_asyncio.fixture
async def plugin(tmp_path):
    agent = MockAgent(tmp_path)
    p = ShellPlugin(agent)
    yield p
    await p._reset()


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")


def _task_id_from(start_msg: str) -> str:
    return next(tok.strip("`") for tok in start_msg.split() if tok.startswith("`task_"))


@pytest.mark.asyncio
async def test_foreground_returns_output_and_streams(plugin):
    result = await plugin.shell(
        command="echo hello && echo world",
        description="Print greeting",
    )
    assert "hello" in result
    assert "world" in result
    streamed = "\n".join(plugin.agent.agent_io.messages)
    assert "Print greeting" in streamed
    assert "hello" in streamed and "world" in streamed
    # Registry should be cleaned up for normal foreground completion.
    assert plugin._background == {}


@pytest.mark.asyncio
async def test_foreground_nonzero_exit_marker(plugin):
    result = await plugin.shell(
        command="echo oops; exit 3",
        description="Force failure",
    )
    assert "oops" in result
    assert "[Process exited with code 3]" in result


@pytest.mark.asyncio
async def test_foreground_stderr_captured(plugin):
    result = await plugin.shell(
        command="echo onerr 1>&2",
        description="Write to stderr",
    )
    assert "[stderr] onerr" in result


@pytest.mark.asyncio
async def test_timeout_clamped_to_max(plugin, monkeypatch, caplog):
    monkeypatch.setattr("arox.plugins.shell.MAX_TIMEOUT_SECONDS", 0)
    with caplog.at_level("WARNING", logger="arox.plugins.shell"):
        result = await plugin.shell(
            command="sleep 30",
            description="Will clamp",
            timeout=9999,
        )
    assert "promoted to background" in result
    assert any("clamping" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_kill_drains_pending_output(plugin):
    # Emit output then sleep; kill while sleeping. Drained output must be
    # captured before kill_shell returns.
    start = await plugin.shell(
        command="echo BEFORE_KILL; sleep 30",
        description="Emit then sleep",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._background[task_id]
    for _ in range(50):
        await asyncio.sleep(0.05)
        if any("BEFORE_KILL" in line for line in bg.captured_lines()):
            break
    await plugin.kill_shell(task_id=task_id, description="stop")
    # After kill returns, drain task should be done (output flushed).
    assert bg.drain_task is None or bg.drain_task.done()


@pytest.mark.asyncio
async def test_head_tail_truncation_marker(plugin, monkeypatch):
    # Force small head/tail buffers so we can exercise the truncation path
    # without producing huge output. The field default_factory reads the
    # module constant at instance-creation time, so monkeypatching the
    # constants is enough.
    monkeypatch.setattr("arox.plugins.shell.HEAD_BUFFER_LINES", 3)
    monkeypatch.setattr("arox.plugins.shell.TAIL_BUFFER_LINES", 3)
    result = await plugin.shell(
        command="for i in 1 2 3 4 5 6 7 8 9 10; do echo line$i; done",
        description="Produce 10 lines",
    )
    assert "line1" in result and "line2" in result and "line3" in result  # head
    assert "line8" in result and "line9" in result and "line10" in result  # tail
    assert "truncated" in result
    assert "line5" not in result  # middle dropped


@pytest.mark.asyncio
async def test_foreground_timeout_promotes_to_background(plugin):
    result = await plugin.shell(
        command="echo early; sleep 30",
        description="Long sleeper",
        timeout=1,
    )
    assert "promoted to background" in result
    assert "early" in result  # partial output included
    task_id = _task_id_from(result)
    bg = plugin._background[task_id]
    # Process should still be alive (not killed).
    assert bg.process.returncode is None
    # Streaming writer cleared; finish notification flag flipped on.
    assert bg.stream_writer is None
    assert bg.notify_on_finish is True


@pytest.mark.asyncio
async def test_background_returns_id_and_polls_output(plugin):
    start = await plugin.shell(
        command="for i in 1 2 3; do echo line$i; sleep 0.05; done",
        description="Emit three lines",
        run_in_background=True,
    )
    assert "Started background task" in start
    task_id = _task_id_from(start)

    bg = plugin._background[task_id]
    assert bg.drain_task is not None
    await asyncio.wait_for(bg.drain_task, timeout=5)

    out = await plugin.shell_state(task_id=task_id, description="poll")
    assert "exit 0" in out
    assert "elapsed:" in out
    assert "line1" in out and "line2" in out and "line3" in out

    again = await plugin.shell_state(task_id=task_id, description="poll again")
    assert "(no new output)" in again
    assert "exit 0" in again


@pytest.mark.asyncio
async def test_shell_state_backoff_grows_per_shell(plugin, monkeypatch):
    # Speed up the base delay so test stays sub-second.
    monkeypatch.setattr("arox.plugins.shell.POLL_BASE_DELAY", 0.05)
    monkeypatch.setattr("arox.plugins.shell.POLL_MAX_DELAY", 1.0)

    start = await plugin.shell(
        command="sleep 30",
        description="Long sleeper",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._background[task_id]

    loop = asyncio.get_event_loop()

    t0 = loop.time()
    await plugin.shell_state(task_id=task_id, description="poll 1")
    d1 = loop.time() - t0
    assert d1 >= 0.04  # ~0.05s

    t1 = loop.time()
    await plugin.shell_state(task_id=task_id, description="poll 2")
    d2 = loop.time() - t1
    assert d2 >= 0.09  # ~0.10s

    t2 = loop.time()
    await plugin.shell_state(task_id=task_id, description="poll 3")
    d3 = loop.time() - t2
    assert d3 >= 0.19  # ~0.20s

    assert d2 > d1 and d3 > d2  # strictly growing
    assert bg.poll_count == 3


@pytest.mark.asyncio
async def test_shell_state_returns_early_when_process_exits(plugin, monkeypatch):
    # Long poll delay so we know early-return is from process exit, not timeout.
    monkeypatch.setattr("arox.plugins.shell.POLL_BASE_DELAY", 10.0)

    start = await plugin.shell(
        command="sleep 0.2; echo done",
        description="Quick task",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._background[task_id]
    assert POLL_BASE_DELAY  # imported but value patched at module level

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    out = await plugin.shell_state(task_id=task_id, description="wait for it")
    waited = loop.time() - t0
    assert waited < 2.0  # returned far earlier than 10s base delay
    assert "exit 0" in out
    assert "done" in out
    assert bg.poll_count == 0  # reset on completion


@pytest.mark.asyncio
async def test_kill_shell_terminates_running_background(plugin):
    start = await plugin.shell(
        command="sleep 30",
        description="Sleep forever",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._background[task_id]
    assert bg.process.returncode is None

    killed = await plugin.kill_shell(task_id=task_id, description="stop it")
    assert "Killed" in killed
    assert "elapsed" in killed
    assert bg.process.returncode is not None
    assert bg.process.returncode in (-signal.SIGTERM, -signal.SIGKILL)


@pytest.mark.asyncio
async def test_kill_unknown_shell(plugin):
    msg = await plugin.kill_shell(task_id="task_nope", description="x")
    assert "Unknown task_id" in msg


@pytest.mark.asyncio
async def test_shell_state_unknown_shell(plugin):
    msg = await plugin.shell_state(task_id="task_nope", description="x")
    assert "Unknown task_id" in msg


@pytest.mark.asyncio
async def test_reset_kills_all_backgrounds(plugin):
    await plugin.shell(command="sleep 30", description="long", run_in_background=True)
    await plugin.shell(command="sleep 30", description="long2", run_in_background=True)
    assert len(plugin._background) == 2

    reset_fn = plugin.agent.get_slot(
        __import__("arox.plugins.slots", fromlist=["AGENT_RESET"]).AGENT_RESET
    )[0]
    await reset_fn()
    assert plugin._background == {}


@pytest.mark.asyncio
async def test_kill_escalates_to_sigkill_when_term_ignored(plugin, monkeypatch):
    py = sys.executable
    cmd = (
        f'exec {py} -c "import signal,sys,time; '
        f"signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"print('READY', flush=True); time.sleep(30)\""
    )
    monkeypatch.setattr("arox.plugins.shell.KILL_GRACE_SECONDS", 0.3)
    start = await plugin.shell(
        command=cmd,
        description="Ignore SIGTERM",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._background[task_id]

    for _ in range(50):
        await asyncio.sleep(0.05)
        if any("READY" in line for line in bg.captured_lines()):
            break
    else:
        pytest.fail("child did not become ready")  # ty: ignore[invalid-argument-type]

    await plugin.kill_shell(task_id=task_id, description="force kill")
    assert bg.process.returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_promoted_shell_eventually_finishes(plugin):
    # Promote then poll until completion, verifying the workflow end-to-end.
    result = await plugin.shell(
        command="echo first; sleep 0.5; echo second",
        description="Mid-length task",
        timeout=1,
    )
    # Either it raced and completed in time, or it was promoted.
    if "promoted to background" not in result:
        assert "first" in result and "second" in result
        return

    task_id = _task_id_from(result)
    bg = plugin._background[task_id]
    await asyncio.wait_for(bg.drain_task, timeout=5)

    final = await plugin.shell_state(task_id=task_id, description="final check")
    assert "exit 0" in final
    assert "second" in final


@pytest.mark.asyncio
async def test_foreground_completion_removes_from_registry(plugin):
    await plugin.shell(command="true", description="noop")
    assert plugin._background == {}


@pytest.mark.asyncio
async def test_background_kills_child_process_tree(plugin):
    # Verify that killing a background shell also kills children it spawned.
    start = await plugin.shell(
        command="sleep 30 & echo CHILD_PID=$!; wait",
        description="Spawn child",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._background[task_id]

    # Wait for the child pid to be reported.
    for _ in range(50):
        await asyncio.sleep(0.05)
        if any("CHILD_PID=" in line for line in bg.captured_lines()):
            break
    else:
        pytest.fail("child pid not reported")  # ty: ignore[invalid-argument-type]

    pid_line = next(line for line in bg.captured_lines() if "CHILD_PID=" in line)
    child_pid = int(pid_line.split("CHILD_PID=", 1)[1].strip())

    await plugin.kill_shell(task_id=task_id, description="terminate")
    await asyncio.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
