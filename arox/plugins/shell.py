import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain

from arox.core.app import get_original_env_copy
from arox.core.plugin import Plugin, tool
from arox.plugins.slots import AGENT_INFO, AGENT_RESET

logger = logging.getLogger(__name__)

HEAD_BUFFER_LINES = 1000
TAIL_BUFFER_LINES = 4000
KILL_GRACE_SECONDS = 5
POLL_BASE_DELAY = 5
POLL_MAX_DELAY = 300
MAX_TIMEOUT_SECONDS = 600
MAX_RENDER_BYTES = 100 * 1024
DRAIN_FLUSH_TIMEOUT = 2.0
DEFAULT_SHELL_UNIX = "/bin/bash"


def _select_shell() -> str:
    if sys.platform == "win32":
        return os.environ.get("COMSPEC", "cmd.exe")
    override = os.environ.get("AROX_SHELL")
    if override:
        return override
    for candidate in (DEFAULT_SHELL_UNIX, "/bin/sh"):
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_SHELL_UNIX


def get_shell_context():
    import platform

    shell_path = _select_shell()
    return {
        "os_info": platform.system(),
        "os_release": platform.release(),
        "shell_type": os.path.basename(shell_path),
    }


@dataclass
class BackgroundShell:
    task_id: str
    command: str
    description: str
    process: asyncio.subprocess.Process
    started_at: float
    head_lines: list[str] = field(default_factory=list)
    tail_lines: deque = field(default_factory=lambda: deque(maxlen=TAIL_BUFFER_LINES))
    total_lines: int = 0
    read_total: int = 0
    exit_code: int | None = None
    finished_at: float | None = None
    drain_task: asyncio.Task | None = None
    poll_count: int = 0
    notify_on_finish: bool = False
    new_output_event: asyncio.Event = field(default_factory=asyncio.Event)

    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    def append_line(self, line: str) -> None:
        if len(self.head_lines) < HEAD_BUFFER_LINES:
            self.head_lines.append(line)
        self.tail_lines.append(line)
        self.total_lines += 1
        self.new_output_event.set()

    def captured_lines(self) -> Iterable[str]:
        """Iterable view of retained lines (head + tail, may overlap)."""
        return chain(self.head_lines, self.tail_lines)

    def render_full(self) -> list[str]:
        """Contiguous view of captured output, with a truncation marker
        if a gap exists between the head and tail buffers."""
        if self.total_lines <= len(self.tail_lines):
            return list(self.tail_lines)
        dropped_from_tail = self.total_lines - len(self.tail_lines)
        if dropped_from_tail <= len(self.head_lines):
            return self.head_lines[:dropped_from_tail] + list(self.tail_lines)
        missing = self.total_lines - len(self.head_lines) - len(self.tail_lines)
        return [
            *self.head_lines,
            f"... ({missing} lines truncated) ...",
            *self.tail_lines,
        ]


class ShellPlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)
        self.workspace = self.agent.workspace.absolute()
        self._background: dict[str, BackgroundShell] = {}

    def on_load(self):
        self.agent.provide_slot(AGENT_INFO, self._get_info)
        self.agent.provide_slot(AGENT_RESET, self._reset)

    def _get_cmd(self, command: str) -> list[str]:
        shell_path = _select_shell()
        if sys.platform == "win32":
            return [shell_path, "/c", command]
        return [shell_path, "-c", command]

    async def _spawn(
        self, command: str, stdin: int | None = asyncio.subprocess.DEVNULL
    ) -> asyncio.subprocess.Process:
        cmd_args = self._get_cmd(command)
        kwargs: dict = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True

        env = get_original_env_copy()
        # Disable color output for self-adapting commands
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"

        return await asyncio.create_subprocess_exec(
            *cmd_args,
            cwd=str(self.workspace),
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            **kwargs,
        )

    def _signal_group(self, process: asyncio.subprocess.Process, sig: int) -> bool:
        if process.returncode is not None:
            return False
        if sys.platform == "win32":
            process.kill()
            return True
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            logger.exception("Failed to signal process group for pid %s", process.pid)
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return False

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """Politely terminate the whole process group, escalating to SIGKILL."""
        if process.returncode is not None:
            return
        self._signal_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=KILL_GRACE_SECONDS)
        except TimeoutError:
            self._signal_group(process, signal.SIGKILL)
            await process.wait()

    async def _flush_drain(self, bg: BackgroundShell) -> None:
        drain = bg.drain_task
        if drain is None or drain.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(drain), timeout=DRAIN_FLUSH_TIMEOUT)
        except (TimeoutError, asyncio.CancelledError):
            pass

    def _allocate_bg(
        self,
        process: asyncio.subprocess.Process,
        command: str,
        description: str,
        *,
        notify_on_finish: bool,
    ) -> BackgroundShell:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        bg = BackgroundShell(
            task_id=task_id,
            command=command,
            description=description,
            process=process,
            started_at=time.monotonic(),
            notify_on_finish=notify_on_finish,
        )
        bg.drain_task = asyncio.create_task(self._drain(bg))
        self._background[task_id] = bg
        return bg

    async def _drain(self, bg: BackgroundShell) -> None:
        async def pump(stream, is_err: bool) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode(errors="replace").rstrip("\r\n")
                display = f"[stderr] {text}" if is_err else text
                bg.append_line(display)
                logger.info("[%s] %s", bg.task_id, display)

        try:
            await asyncio.gather(
                pump(bg.process.stdout, False),
                pump(bg.process.stderr, True),
            )
            await bg.process.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Task %s drain failed", bg.task_id)
        finally:
            bg.exit_code = bg.process.returncode
            bg.finished_at = time.monotonic()
            logger.info(
                "Task %s finished: %s (exit %s, elapsed %.1fs)",
                bg.task_id,
                bg.description,
                bg.exit_code,
                bg.elapsed(),
            )
            if bg.notify_on_finish:
                try:
                    await self.agent.agent_io.send(
                        f"[bg {bg.task_id}] task finished "
                        f"(exit {bg.exit_code}, elapsed {bg.elapsed():.1f}s)"
                    )
                except Exception:
                    pass

    @tool(dynamic_context=get_shell_context)
    async def shell(
        self,
        command: str,
        description: str = "",
        timeout: int | None = 100,
        run_in_background: bool = False,
    ) -> str:
        """Run a shell command and return its output.

        Environment Info:
        - OS: {{ os_info }} {{ os_release }}
        - Shell: {{ shell_type }}

        Rules
            1. For searching code, use `rg` or `ast-grep`.
            2. If a command requires stdin input (interactive commands), you MUST set `run_in_background=True`. Foreground commands have stdin attached to /dev/null and will fail if they try to read input. For background tasks, use `shell_state` to see the prompt and `shell_input` to provide the required input.
            3. The command runs via `{{ shell_type }} -c`, so quoting matters:
               - Wrap literal text in single quotes to disable $ expansion:
                     echo 'literal $HOME and `cmd`'
               - To include a single quote inside single quotes, close/reopen:
                     echo 'it'\\''s here'
               - For multi-line literals prefer heredoc with a quoted delimiter:
                     cat <<'EOF'
                     line one
                     line two
                     EOF
            4. stderr lines are tagged with a leading `[stderr]` so they are
               distinguishable from stdout. Exit code is shown when non-zero.
            5. Chain with `&&` to stop on the first failure; use `;` only when
               every step should run regardless of previous exit codes.

        Long-running commands
            If a foreground command does not finish within `timeout` seconds,
            it is NOT killed — it is promoted to a background task and the
            returned message contains a `task_id` plus partial output. Poll
            it with `shell_state(task_id=...)` or kill it with
            `kill_shell(task_id=...)`.

            Set `run_in_background=True` explicitly when you already know the
            command is long-lived (dev server, watcher, multi-minute build).

        Examples
            command: "ls -la | rg staff"
            description: "List files matching 'staff'"

        Args:
            command: The shell command to execute.
            description: One short sentence describing what this command does
                (e.g., "Run unit tests"). Shown to the user and used in logs.
            timeout: Seconds to wait in the foreground before promoting to
                background (default 100, hard cap 600). Ignored when
                `run_in_background=True`.
            run_in_background: If True, start detached and return a task id
                immediately.

        Returns:
            Foreground that finished in time: combined stdout/stderr (with a
            head+tail truncation marker if very large) and an exit-code marker
            on non-zero exit.
            Foreground that timed out: promotion notice with task_id and
            partial output.
            Background: a task id and usage hint.
        """
        if timeout is not None and timeout > MAX_TIMEOUT_SECONDS:
            logger.warning(
                "Requested timeout %ss exceeds cap %ss; clamping",
                timeout,
                MAX_TIMEOUT_SECONDS,
            )
            timeout = MAX_TIMEOUT_SECONDS

        logger.info("Executing shell command (%s): %s", description, command)
        try:
            stdin = (
                asyncio.subprocess.PIPE
                if run_in_background
                else asyncio.subprocess.DEVNULL
            )
            process = await self._spawn(command, stdin=stdin)
        except Exception as e:
            return f"Error spawning command: {e!s}"

        if run_in_background:
            bg = self._allocate_bg(
                process,
                command,
                description,
                notify_on_finish=True,
            )
            await self.agent.agent_io.send(
                f"[bg {bg.task_id}] task started: {description}  (pid {process.pid})"
            )
            return (
                f"Started background task `{bg.task_id}` "
                f"(pid {process.pid}): {description}\n"
                f'- Poll output: shell_state(task_id="{bg.task_id}")\n'
                f'- Terminate:   kill_shell(task_id="{bg.task_id}")'
            )

        # Foreground: wait for completion or promote to background on timeout.
        bg = self._allocate_bg(
            process,
            command,
            description,
            notify_on_finish=False,
        )
        drain = bg.drain_task
        assert drain is not None
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(drain), timeout=timeout)
        except TimeoutError:
            timed_out = True

        if timed_out:
            bg.notify_on_finish = True
            tail_lines = list(bg.tail_lines)[-50:]
            tail = "\n".join(tail_lines)
            logger.info(
                "Task %s promoted to background after %ss: %s",
                bg.task_id,
                timeout,
                bg.description,
            )
            msg = (
                f"Command did not finish within {timeout}s — promoted to "
                f"background task `{bg.task_id}` (still running, elapsed "
                f"{bg.elapsed():.1f}s).\n"
                f'- Poll:      shell_state(task_id="{bg.task_id}")\n'
                f'- Terminate: kill_shell(task_id="{bg.task_id}")'
            )
            await self.agent.agent_io.send(
                f"[{bg.task_id}] promoted to background after {timeout}s"
            )
            if tail:
                return f"{msg}\n--- partial output (last 50 lines) ---\n{tail}"
            return msg

        output = self._render_completed(bg)
        self._background.pop(bg.task_id, None)
        return output

    def _render_completed(self, bg: BackgroundShell) -> str:
        lines = bg.render_full()
        text = "\n".join(lines)
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_RENDER_BYTES:
            half = MAX_RENDER_BYTES // 2
            head_text = encoded[:half].decode("utf-8", errors="replace")
            tail_text = encoded[-half:].decode("utf-8", errors="replace")
            text = (
                f"{head_text}\n"
                f"... (output truncated by size, {bg.total_lines} total lines) ...\n"
                f"{tail_text}"
            )
        if bg.exit_code != 0:
            if text:
                text += "\n"
            text += f"[Process exited with code {bg.exit_code}]"
        return text

    @tool()
    async def shell_state(self, task_id: str, description: str = "") -> str:
        """Check on a background (or promoted) task. Waits briefly (with
        per-task exponential backoff: 20s, 40s, 80s, ..., capped at 300s)
        so the model doesn't busy-poll, then returns new output, run status,
        and total elapsed time.

        The wait returns early if the process exits in the meantime. The
        backoff resets once the process reaches a terminal state.

        Args:
            task_id: The id returned by `shell(...)`.
            description: One short sentence on why you're checking
                (e.g., "Wait for dev server to bind port").
        """
        logger.info("Polling task %s (%s)", task_id, description)
        bg = self._background.get(task_id)
        if bg is None:
            return f"Unknown task_id: {task_id}"

        drain = bg.drain_task
        new_count = bg.total_lines - bg.read_total

        # Block only if there are no new lines and the process is still running.
        if new_count == 0 and bg.exit_code is None and drain is not None:
            delay = min(POLL_BASE_DELAY * (2**bg.poll_count), POLL_MAX_DELAY)
            bg.poll_count += 1
            bg.new_output_event.clear()
            try:
                wait_event = asyncio.create_task(bg.new_output_event.wait())
                done, pending = await asyncio.wait(
                    [asyncio.shield(drain), wait_event],
                    timeout=delay,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if wait_event in pending:
                    wait_event.cancel()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            # Recalculate after the potential wait
            new_count = bg.total_lines - bg.read_total

        # Reset backoff if we actually received output
        if new_count > 0:
            bg.poll_count = 0

        if new_count <= 0:
            new_lines: list[str] = []
        elif new_count <= len(bg.tail_lines):
            new_lines = list(bg.tail_lines)[-new_count:]
        else:
            dropped = new_count - len(bg.tail_lines)
            new_lines = [
                f"... ({dropped} lines dropped before this slice) ...",
                *bg.tail_lines,
            ]
        bg.read_total = bg.total_lines

        if bg.exit_code is not None:
            status = f"exit {bg.exit_code}"
            bg.poll_count = 0
        else:
            status = "running"

        header = f"[{task_id}] status: {status} | elapsed: {bg.elapsed():.1f}s"

        if not new_lines:
            return f"{header}\n(no new output)"

        body = "\n".join(new_lines)
        encoded = body.encode("utf-8")
        if len(encoded) > MAX_RENDER_BYTES:
            half = MAX_RENDER_BYTES // 2
            head_text = encoded[:half].decode("utf-8", errors="replace")
            tail_text = encoded[-half:].decode("utf-8", errors="replace")
            body = (
                f"{head_text}\n"
                f"... (poll output truncated by size, {len(new_lines)} new lines) ...\n"
                f"{tail_text}"
            )
        return f"{header}\n{body}"

    @tool()
    async def shell_input(self, task_id: str, text: str, description: str = "") -> str:
        """Send text to the stdin of a running background task.

        Args:
            task_id: The id of the running task.
            text: The text to send (e.g., "y\\n").
            description: Why you are sending this input.
        """
        logger.info("Sending input to task %s (%s): %r", task_id, description, text)
        bg = self._background.get(task_id)
        if bg is None:
            return f"Unknown task_id: {task_id}"
        if bg.exit_code is not None:
            return f"Task {task_id} has already exited with code {bg.exit_code}."

        if bg.process.stdin is None:
            return f"Task {task_id} does not support stdin."

        try:
            if not text.endswith("\n"):
                logger.warning(
                    "shell_input received text without newline, appending \\n automatically."
                )
                text += "\n"
            bg.process.stdin.write(text.encode())
            await bg.process.stdin.drain()
            return f"Sent input to task {task_id}."
        except Exception as e:
            logger.exception("Failed to send input to task %s", task_id)
            return f"Failed to send input: {e!s}"

    @tool()
    async def kill_shell(self, task_id: str, description: str = "") -> str:
        """Terminate a background (or promoted) task. Sends SIGTERM to the
        whole process group, escalating to SIGKILL after a short grace period.

        Args:
            task_id: The id returned by `shell(...)`.
            description: One short sentence on why (e.g., "Tests finished").
        """
        logger.info("Killing task %s (%s)", task_id, description)
        bg = self._background.get(task_id)
        if bg is None:
            return f"Unknown task_id: {task_id}"
        if bg.process.returncode is not None:
            return (
                f"Task {task_id} already exited with code "
                f"{bg.process.returncode} (elapsed {bg.elapsed():.1f}s)"
            )
        await self._terminate(bg.process)
        await self._flush_drain(bg)
        logger.info(
            "Task %s killed: %s (exit %s, elapsed %.1fs)",
            bg.task_id,
            bg.description,
            bg.process.returncode,
            bg.elapsed(),
        )
        return (
            f"Killed task {task_id} "
            f"(exit code {bg.process.returncode}, elapsed {bg.elapsed():.1f}s)"
        )

    async def _reset(self) -> None:
        for task_id in list(self._background):
            bg = self._background.pop(task_id)
            if bg.process.returncode is None:
                await self._terminate(bg.process)
            if bg.drain_task and not bg.drain_task.done():
                bg.drain_task.cancel()

    async def _get_info(self) -> str:
        if not self._background:
            return ""
        lines = ["", f"Background tasks ({len(self._background)}):"]
        for task_id, bg in self._background.items():
            if bg.exit_code is not None:
                state = f"exit {bg.exit_code}"
            else:
                state = "running"
            lines.append(
                f"  - {task_id} [{state} {bg.elapsed():.0f}s] {bg.description}"
            )
        return "\n".join(lines)
