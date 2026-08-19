import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from arox.core.background import BackgroundTaskBroker
from arox.plugins.shell import POLL_BASE_DELAY, ShellPlugin


class CaptureIO:
    def __init__(self):
        self.messages: list[str] = []

    async def send(self, msg):
        self.messages.append(msg)


class MockAgent:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.agent_ep = CaptureIO()
        self._slots: dict = {}
        self.background_tasks = BackgroundTaskBroker()
        from arox.core.config import Config
        from arox.core.session import AgentSession, FileSessionStore, SessionManager

        self.config = Config()
        store = FileSessionStore()
        store.base_dir = workspace / "sessions"
        manager = SessionManager(store)
        self.session = AgentSession(agent_name="test", path=["test-session"])
        self.session.manager = manager

    def provide_slot(self, slot, provider):
        self._slots.setdefault(slot, []).append(provider)

    async def invoke_slot(self, slot, *args, **kwargs):
        providers = self._slots.get(slot, [])
        from arox.core.slot import ResultAggregator

        match slot.aggregator:
            case ResultAggregator.DISCARD:
                for handler in providers:
                    result = handler(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        await result
                return None
            case ResultAggregator.FIRST:
                if not providers:
                    return None
                result = providers[0](*args, **kwargs)
                return await result if asyncio.iscoroutine(result) else result
            case ResultAggregator.LIST:
                results = []
                for handler in providers:
                    result = handler(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                    results.append(result)
                return results

    def register(self, plugin):
        plugin.on_load()
        return plugin


@pytest_asyncio.fixture
async def plugin(tmp_path):
    agent = MockAgent(tmp_path)
    p = agent.register(ShellPlugin(agent))
    yield p
    await p.on_stop()


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")


def _task_id_from(start_msg: str) -> str:
    return next(tok.strip("`") for tok in start_msg.split() if tok.startswith("`task_"))


def _output_path(result: str) -> Path:
    prefixes = ("Full output saved to: ", "Full output is being saved to: ")
    for line in result.splitlines():
        for prefix in prefixes:
            if line.startswith(prefix):
                return Path(line.removeprefix(prefix))
        if line.startswith("Captured output: "):
            return Path(line.removeprefix("Captured output: "))
    raise AssertionError("output path missing from shell result")


@pytest.mark.asyncio
async def test_foreground_returns_output_and_removes_complete_log(plugin):
    result = await plugin.shell(
        command="echo hello && echo world",
        description="Print greeting",
    )
    assert "hello" in result
    assert "world" in result
    assert "Full output" not in result
    assert "Captured output" not in result
    assert list(plugin._session_output_dir().iterdir()) == []
    assert plugin._tasks == {}


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
    bg = plugin._tasks[task_id]
    for _ in range(50):
        await asyncio.sleep(0.01)
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
    assert "line5" not in result  # middle omitted from the response

    output_path = _output_path(result)
    assert output_path.read_text().splitlines() == [f"line{i}" for i in range(1, 11)]


@pytest.mark.asyncio
async def test_render_limit_only_applies_to_output_data(plugin, monkeypatch):
    monkeypatch.setattr("arox.plugins.shell.MAX_RENDER_BYTES", 40)
    result = await plugin.shell(
        command="printf 'abcdefghij\\n%.0s' 1 2 3 4 5 6 7 8",
        description="Produce wide output",
    )

    output_section = result.split("\nFull output saved to:", 1)[0]
    output_lines = [
        line for line in output_section.splitlines() if "output truncated:" not in line
    ]
    assert sum(len(line.encode()) + 1 for line in output_lines) <= 40
    assert "output data limited to 10 KiB" in result
    assert "Full output saved to:" in result


@pytest.mark.asyncio
async def test_background_poll_summarizes_but_file_keeps_all_output(
    plugin, monkeypatch
):
    monkeypatch.setattr("arox.plugins.shell.HEAD_BUFFER_LINES", 2)
    monkeypatch.setattr("arox.plugins.shell.TAIL_BUFFER_LINES", 2)
    start = await plugin.shell(
        command="for i in 1 2 3 4 5 6 7 8; do echo line$i; done",
        description="Produce background output",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    task = plugin._tasks[task_id]
    await asyncio.wait_for(task.drain_task, timeout=5)

    result = await plugin.shell_state(task_id=task_id, description="poll")
    assert "line1" in result and "line2" in result
    assert "line7" in result and "line8" in result
    assert "line4" not in result
    assert "output truncated" in result
    assert _output_path(result).read_text().splitlines() == [
        f"line{i}" for i in range(1, 9)
    ]


@pytest.mark.asyncio
async def test_output_file_limit_does_not_stop_drain(plugin, monkeypatch):
    monkeypatch.setattr("arox.plugins.shell.MAX_OUTPUT_FILE_BYTES", 20)
    monkeypatch.setattr("arox.plugins.shell.MAX_RENDER_BYTES", 20)
    result = await plugin.shell(
        command="printf '1234567890\\n1234567890\\nAFTER_LIMIT\\n'",
        description="Exceed output file limit",
    )

    output_path = _output_path(result)
    assert output_path.stat().st_size <= 20
    assert "1 lines omitted" in result
    assert "reached the 10 MiB limit" in result


@pytest.mark.asyncio
async def test_session_delete_removes_shell_output(plugin, monkeypatch):
    monkeypatch.setattr("arox.plugins.shell.HEAD_BUFFER_LINES", 1)
    monkeypatch.setattr("arox.plugins.shell.TAIL_BUFFER_LINES", 1)
    result = await plugin.shell(
        command="printf 'first\\nmiddle\\nlast\\n'", description="Save output"
    )
    output_path = _output_path(result)
    assert output_path.exists()

    session = plugin.runtime.session
    await session.manager.session_store.delete_session(session.path)
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_foreground_timeout_promotes_to_background(plugin):
    result = await plugin.shell(
        command="echo early; sleep 30",
        description="Long sleeper",
        timeout=0.1,
    )
    assert "promoted to background" in result
    assert "early" in result  # partial output included
    task_id = _task_id_from(result)
    bg = plugin._tasks[task_id]
    # Process should still be alive (not killed).
    assert bg.process.returncode is None
    # Finish notification flag flipped on.
    assert bg.notify_on_finish is True
    assert "can poll for output at any time" in result
    assert "will also be notified when the command finishes" in result


@pytest.mark.asyncio
async def test_background_returns_id_and_polls_output(plugin):
    start = await plugin.shell(
        command="for i in 1 2 3; do echo line$i; sleep 0.01; done",
        description="Emit three lines",
        run_in_background=True,
    )
    assert "Started background task" in start
    assert "can poll for output at any time" in start
    assert "will also be notified when the command finishes" in start
    task_id = _task_id_from(start)

    bg = plugin._tasks[task_id]
    assert bg.drain_task is not None
    await asyncio.wait_for(bg.drain_task, timeout=5)

    notifications = plugin.runtime.background_tasks.drain_notices()
    assert len(notifications) == 1
    notification = notifications[0]
    assert task_id in notification
    assert "Description: Emit three lines" in notification
    assert f'shell_state(task_id="{task_id}")' in notification
    assert "line1" not in notification

    out = await plugin.shell_state(task_id=task_id, description="poll")
    assert "exit 0" in out
    assert "elapsed:" in out
    assert "line1" in out and "line2" in out and "line3" in out

    again = await plugin.shell_state(task_id=task_id, description="poll again")
    assert "(no new output)" in again
    assert "exit 0" in again


@pytest.mark.asyncio
async def test_terminal_shell_state_suppresses_completion_notice(plugin):
    start = await plugin.shell(
        command="true",
        description="Finish quietly",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    await plugin._tasks[task_id].drain_task

    state = await plugin.shell_state(task_id=task_id)

    assert "exit 0" in state
    assert plugin.runtime.background_tasks.drain_notices() == []


@pytest.mark.asyncio
async def test_shell_state_backoff_grows_per_shell(plugin, monkeypatch):
    # Speed up the base delay so test stays sub-second.
    monkeypatch.setattr("arox.plugins.shell.POLL_BASE_DELAY", 0.01)
    monkeypatch.setattr("arox.plugins.shell.POLL_MAX_DELAY", 1.0)

    start = await plugin.shell(
        command="sleep 30",
        description="Long sleeper",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._tasks[task_id]

    loop = asyncio.get_event_loop()

    t0 = loop.time()
    await plugin.shell_state(task_id=task_id, description="poll 1")
    d1 = loop.time() - t0
    assert d1 >= 0.008

    t1 = loop.time()
    await plugin.shell_state(task_id=task_id, description="poll 2")
    d2 = loop.time() - t1
    assert d2 >= 0.018

    t2 = loop.time()
    await plugin.shell_state(task_id=task_id, description="poll 3")
    d3 = loop.time() - t2
    assert d3 >= 0.038

    assert d2 > d1 and d3 > d2  # strictly growing
    assert bg.poll_count == 3


@pytest.mark.asyncio
async def test_shell_state_returns_early_when_process_exits(plugin, monkeypatch):
    # Long poll delay so we know early-return is from process exit, not timeout.
    monkeypatch.setattr("arox.plugins.shell.POLL_BASE_DELAY", 10.0)

    start = await plugin.shell(
        command="sleep 0.05",
        description="Quick task",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._tasks[task_id]
    assert POLL_BASE_DELAY  # imported but value patched at module level

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    out = await plugin.shell_state(task_id=task_id, description="wait for it")
    waited = loop.time() - t0
    assert waited < 2.0  # returned far earlier than 10s base delay
    assert "exit 0" in out
    assert bg.poll_count == 0  # reset on completion


@pytest.mark.asyncio
async def test_kill_shell_terminates_running_background(plugin):
    start = await plugin.shell(
        command="sleep 30",
        description="Sleep forever",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._tasks[task_id]
    assert bg.process.returncode is None

    killed = await plugin.kill_shell(task_id=task_id, description="stop it")
    assert "Killed" in killed
    assert "elapsed" in killed
    assert bg.process.returncode is not None
    assert bg.process.returncode in (-signal.SIGTERM, -signal.SIGKILL)


@pytest.mark.asyncio
async def test_shell_input(plugin):
    # Use a command that reads from stdin
    py = sys.executable
    cmd = f"{py} -c \"import sys; line = sys.stdin.readline(); print(f'GOT:{{line.strip()}}', flush=True)\""
    start = await plugin.shell(
        command=cmd,
        description="Read from stdin",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._tasks[task_id]

    # Send input
    res = await plugin.shell_input(
        task_id=task_id, text="hello\n", description="say hello"
    )
    assert "Sent input" in res

    # Wait for completion
    await asyncio.wait_for(bg.drain_task, timeout=5)

    # Check output
    out = await plugin.shell_state(task_id=task_id, description="check output")
    assert "GOT:hello" in out
    assert "exit 0" in out


@pytest.mark.asyncio
async def test_kill_unknown_shell(plugin):
    msg = await plugin.kill_shell(task_id="task_nope", description="x")
    assert "Unknown task_id" in msg


@pytest.mark.asyncio
async def test_shell_state_unknown_shell(plugin):
    msg = await plugin.shell_state(task_id="task_nope", description="x")
    assert "Unknown task_id" in msg


@pytest.mark.asyncio
async def test_on_stop_kills_all_backgrounds(plugin):
    await plugin.shell(command="sleep 30", description="long", run_in_background=True)
    await plugin.shell(command="sleep 30", description="long2", run_in_background=True)
    assert len(plugin._tasks) == 2

    await plugin.on_stop()
    assert plugin._tasks == {}


@pytest.mark.asyncio
async def test_kill_escalates_to_sigkill_when_term_ignored(plugin, monkeypatch):
    py = sys.executable
    cmd = (
        f'exec {py} -c "import signal,sys,time; '
        f"signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"print('READY', flush=True); time.sleep(30)\""
    )
    monkeypatch.setattr("arox.plugins.shell.KILL_GRACE_SECONDS", 0.05)
    start = await plugin.shell(
        command=cmd,
        description="Ignore SIGTERM",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._tasks[task_id]

    for _ in range(50):
        await asyncio.sleep(0.01)
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
        command="echo first; sleep 0.1; echo second",
        description="Mid-length task",
        timeout=0.05,
    )
    # Either it raced and completed in time, or it was promoted.
    if "promoted to background" not in result:
        assert "first" in result and "second" in result
        return

    task_id = _task_id_from(result)
    bg = plugin._tasks[task_id]
    await asyncio.wait_for(bg.drain_task, timeout=5)

    final = await plugin.shell_state(task_id=task_id, description="final check")
    assert "exit 0" in final
    assert "second" in final


@pytest.mark.asyncio
async def test_foreground_completion_removes_from_registry(plugin):
    await plugin.shell(command="true", description="noop")
    assert plugin._tasks == {}


@pytest.mark.asyncio
async def test_background_kills_child_process_tree(plugin):
    # Verify that killing a background shell also kills children it spawned.
    start = await plugin.shell(
        command="sleep 30 & echo CHILD_PID=$!; wait",
        description="Spawn child",
        run_in_background=True,
    )
    task_id = _task_id_from(start)
    bg = plugin._tasks[task_id]

    # Wait for the child pid to be reported.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if any("CHILD_PID=" in line for line in bg.captured_lines()):
            break
    else:
        pytest.fail("child pid not reported")  # ty: ignore[invalid-argument-type]

    pid_line = next(line for line in bg.captured_lines() if "CHILD_PID=" in line)
    child_pid = int(pid_line.split("CHILD_PID=", 1)[1].strip())

    await plugin.kill_shell(task_id=task_id, description="terminate")

    for _ in range(20):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Child process was not killed")  # ty: ignore[invalid-argument-type]


@pytest.mark.asyncio
async def test_env_vars_restoration(plugin, monkeypatch):
    # Setup: mock app config with some env vars
    plugin.runtime.config.app.env_vars = {"TEST_VAR": "arox_value"}
    plugin.runtime.config.app.api_keys = {"test_provider": "arox_key"}

    # Mock os.environ and original env
    monkeypatch.setenv("TEST_VAR", "arox_value")
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "arox_key")
    monkeypatch.setenv("OTHER_VAR", "keep_me")

    # We need to mock get_original_env_copy to return a state where TEST_VAR was different or missing
    from arox.core.app import _ORIGINAL_ENV

    original_env_mock = _ORIGINAL_ENV.copy()
    original_env_mock["TEST_VAR"] = "original_value"
    original_env_mock["OTHER_VAR"] = "keep_me"
    # TEST_PROVIDER_API_KEY is missing from original_env_mock

    monkeypatch.setattr(
        "arox.plugins.shell.get_original_env_copy", lambda: original_env_mock
    )

    result = await plugin.shell(
        command="echo TEST_VAR=$TEST_VAR && echo TEST_PROVIDER_API_KEY=$TEST_PROVIDER_API_KEY && echo OTHER_VAR=$OTHER_VAR",
        description="Check env vars",
    )

    assert "TEST_VAR=original_value" in result
    assert "TEST_PROVIDER_API_KEY=" in result
    assert "TEST_PROVIDER_API_KEY=arox_key" not in result
    assert "OTHER_VAR=keep_me" in result
