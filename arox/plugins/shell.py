import asyncio
import contextlib
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
from pathlib import Path
from typing import BinaryIO

from arox.core.app import get_original_env_copy
from arox.core.plugin import Plugin, tool
from arox.plugins.slots import AGENT_INFO

logger = logging.getLogger(__name__)

HEAD_BUFFER_LINES = 50
TAIL_BUFFER_LINES = 150
KILL_GRACE_SECONDS = 5
POLL_BASE_DELAY = 5
POLL_MAX_DELAY = 300
MAX_TIMEOUT_SECONDS = 600
MAX_RENDER_BYTES = 10 * 1024
MAX_OUTPUT_FILE_BYTES = 10 * 1024 * 1024
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
class RenderedOutput:
    text: str
    omitted_lines: int = 0
    truncated_by_size: bool = False

    @property
    def truncated(self) -> bool:
        return self.omitted_lines > 0 or self.truncated_by_size


@dataclass
class ShellTask:
    task_id: str
    command: str
    description: str
    process: asyncio.subprocess.Process
    started_at: float
    output_path: Path
    output_file: BinaryIO
    head_lines: list[str] = field(default_factory=list)
    tail_lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=TAIL_BUFFER_LINES)
    )
    unread_head_lines: list[str] = field(default_factory=list)
    unread_tail_lines: deque[str] = field(
        default_factory=lambda: deque(maxlen=TAIL_BUFFER_LINES)
    )
    total_lines: int = 0
    unread_lines: int = 0
    output_file_bytes: int = 0
    output_file_limited: bool = False
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
        if len(self.unread_head_lines) < HEAD_BUFFER_LINES:
            self.unread_head_lines.append(line)
        self.unread_tail_lines.append(line)
        self.total_lines += 1
        self.unread_lines += 1
        self._write_output(line)
        self.new_output_event.set()

    def _write_output(self, line: str) -> None:
        if self.output_file_limited:
            return
        data = f"{line}\n".encode()
        remaining = MAX_OUTPUT_FILE_BYTES - self.output_file_bytes
        if len(data) <= remaining:
            self.output_file.write(data)
            self.output_file_bytes += len(data)
            return

        if remaining > 0:
            prefix = data[:remaining].decode(errors="ignore").encode()
            self.output_file.write(prefix)
            self.output_file_bytes += len(prefix)
        self.output_file_limited = True

    def captured_lines(self) -> Iterable[str]:
        """Iterable view of retained lines (head + tail, may overlap)."""
        return chain(self.head_lines, self.tail_lines)

    def take_unread(self) -> tuple[list[str], list[str], int]:
        head = self.unread_head_lines
        tail = list(self.unread_tail_lines)
        total = self.unread_lines
        self.unread_head_lines = []
        self.unread_tail_lines = deque(maxlen=TAIL_BUFFER_LINES)
        self.unread_lines = 0
        return head, tail, total


class ShellPlugin(Plugin):
    def __init__(self, runtime):
        super().__init__(runtime)
        self.workspace = self.runtime.workspace.absolute()
        self._tasks: dict[str, ShellTask] = {}

    def on_load(self):
        self.runtime.provide_slot(AGENT_INFO, self._get_info)

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

    async def _flush_drain(self, task: ShellTask) -> None:
        drain = task.drain_task
        if drain is None or drain.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(drain), timeout=DRAIN_FLUSH_TIMEOUT)
        except (TimeoutError, asyncio.CancelledError):
            pass

    def _session_output_dir(self) -> Path:
        session = self.runtime.session
        manager = session.manager
        if manager is None:
            raise RuntimeError("Agent session is not attached to a session manager")
        output_dir = manager.session_store.session_dir(session.path) / "shell"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _register_task(
        self,
        process: asyncio.subprocess.Process,
        command: str,
        description: str,
        *,
        notify_on_finish: bool,
    ) -> ShellTask:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        output_path = self._session_output_dir() / f"{task_id}.log"
        output_file = output_path.open("wb", buffering=0)
        task = ShellTask(
            task_id=task_id,
            command=command,
            description=description,
            process=process,
            started_at=time.monotonic(),
            output_path=output_path,
            output_file=output_file,
            notify_on_finish=notify_on_finish,
        )
        task.drain_task = asyncio.create_task(self._drain(task))
        self._tasks[task_id] = task
        return task

    async def _drain(self, task: ShellTask) -> None:
        async def pump(stream, is_err: bool) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode(errors="replace").rstrip("\r\n")
                display = f"[stderr] {text}" if is_err else text
                task.append_line(display)
                logger.info("[%s] %s", task.task_id, display)

        try:
            await asyncio.gather(
                pump(task.process.stdout, False),
                pump(task.process.stderr, True),
            )
            await task.process.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Task %s drain failed", task.task_id)
        finally:
            task.output_file.close()
            task.exit_code = task.process.returncode
            task.finished_at = time.monotonic()
            logger.info(
                "Task %s finished: %s (exit %s, elapsed %.1fs)",
                task.task_id,
                task.description,
                task.exit_code,
                task.elapsed(),
            )
            if task.notify_on_finish:
                try:
                    await self.runtime.agent_io.send(
                        f"[bg {task.task_id}] task finished "
                        f"(exit {task.exit_code}, elapsed {task.elapsed():.1f}s)"
                    )
                except Exception:
                    pass

    @staticmethod
    def _retained_parts(
        head_lines: list[str], tail_lines: list[str], total_lines: int
    ) -> tuple[list[str], list[str]]:
        if total_lines <= 0:
            return [], []
        if total_lines <= len(tail_lines):
            retained = tail_lines[-total_lines:]
        else:
            before_tail = total_lines - len(tail_lines)
            if before_tail <= len(head_lines):
                retained = head_lines[:before_tail] + tail_lines
            else:
                return head_lines, tail_lines
        head_count = min(HEAD_BUFFER_LINES, len(retained))
        return retained[:head_count], retained[head_count:]

    @staticmethod
    def _data_size(lines: list[str]) -> int:
        return sum(len(line.encode()) + 1 for line in lines)

    @staticmethod
    def _fit_lines(
        lines: list[str], budget: int, *, from_end: bool
    ) -> tuple[list[str], int, bool]:
        if not lines or budget <= 0:
            return [], 0, bool(lines)

        selected: list[str] = []
        used = 0
        truncated = False
        source = reversed(lines) if from_end else iter(lines)
        for line in source:
            encoded = line.encode()
            cost = len(encoded) + 1
            remaining = budget - used
            if cost <= remaining:
                selected.append(line)
                used += cost
                continue
            if remaining > 1:
                available = remaining - 1
                if from_end:
                    fragment = encoded[-available:].decode(errors="ignore")
                else:
                    fragment = encoded[:available].decode(errors="ignore")
                if fragment:
                    selected.append(fragment)
                    used += len(fragment.encode()) + 1
            truncated = True
            break

        if from_end:
            selected.reverse()
        if len(selected) < len(lines):
            truncated = True
        return selected, used, truncated

    def _render_output(
        self, head_lines: list[str], tail_lines: list[str], total_lines: int
    ) -> RenderedOutput:
        head, tail = self._retained_parts(head_lines, tail_lines, total_lines)
        head_size = self._data_size(head)
        tail_size = self._data_size(tail)
        if head_size + tail_size <= MAX_RENDER_BYTES:
            rendered_head = head
            rendered_tail = tail
            head_count = len(head)
            tail_count = len(tail)
            truncated_by_size = False
        else:
            head_budget = min(head_size, MAX_RENDER_BYTES // 4)
            tail_budget = min(tail_size, MAX_RENDER_BYTES - head_budget)
            head_budget = min(head_size, MAX_RENDER_BYTES - tail_budget)
            tail_budget = min(tail_size, MAX_RENDER_BYTES - head_budget)
            rendered_head, _, head_truncated = self._fit_lines(
                head, head_budget, from_end=False
            )
            rendered_tail, _, tail_truncated = self._fit_lines(
                tail, tail_budget, from_end=True
            )
            head_count = len(rendered_head)
            tail_count = len(rendered_tail)
            truncated_by_size = head_truncated or tail_truncated

        omitted_lines = max(total_lines - head_count - tail_count, 0)
        body = list(rendered_head)
        if omitted_lines or truncated_by_size:
            details = []
            if omitted_lines:
                details.append(f"{omitted_lines} lines omitted")
            if truncated_by_size:
                details.append("output data limited to 10 KiB")
            body.append(f"... (output truncated: {'; '.join(details)}) ...")
        body.extend(rendered_tail)
        return RenderedOutput(
            text="\n".join(body),
            omitted_lines=omitted_lines,
            truncated_by_size=truncated_by_size,
        )

    @staticmethod
    def _remove_output_file(task: ShellTask) -> bool:
        try:
            task.output_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove output file for task %s", task.task_id)
            return False
        return True

    @staticmethod
    def _output_notice(task: ShellTask) -> str:
        if task.output_file_limited:
            return (
                f"Captured output: {task.output_path}\n"
                "The file reached the 10 MiB limit; later output was consumed "
                "but not saved."
            )
        if task.exit_code is None:
            return f"Full output is being saved to: {task.output_path}"
        return f"Full output saved to: {task.output_path}"

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
            1. Prefer `rg` for code search. Use `--no-ignore-*` flags to override default ignore rules.
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

        Output handling
            Each response shows at most 50 head lines and 150 tail lines, with
            command output data limited to 10 KiB. Complete captured output is
            written to the session's shell directory, capped at 10 MiB per
            task. A completed foreground task removes this file when its full
            output fits in the response; truncated and background output files
            remain available for inspection.

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
            Foreground that finished in time: combined stdout/stderr summary,
            exit code when non-zero, and the captured-output path when the
            response is truncated.
            Foreground that timed out: promotion notice, output summary, task
            id, and the captured-output path.
            Background: a task id, output path, and usage hint.
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

        try:
            task = self._register_task(
                process,
                command,
                description,
                notify_on_finish=run_in_background,
            )
        except Exception as e:
            await self._terminate(process)
            return f"Error creating command output file: {e!s}"

        if run_in_background:
            await self.runtime.agent_io.send(
                f"[bg {task.task_id}] task started: {description}  (pid {process.pid})"
            )
            return (
                f"Started background task `{task.task_id}` "
                f"(pid {process.pid}): {description}\n"
                f"{self._output_notice(task)}\n"
                f'- Poll output: shell_state(task_id="{task.task_id}")\n'
                f'- Terminate:   kill_shell(task_id="{task.task_id}")'
            )

        drain = task.drain_task
        assert drain is not None
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(drain), timeout=timeout)
        except TimeoutError:
            timed_out = True

        if timed_out:
            task.notify_on_finish = True
            rendered = self._render_output(
                task.head_lines, list(task.tail_lines), task.total_lines
            )
            logger.info(
                "Task %s promoted to background after %ss: %s",
                task.task_id,
                timeout,
                task.description,
            )
            parts = [
                f"Command did not finish within {timeout}s — promoted to "
                f"background task `{task.task_id}` (still running, elapsed "
                f"{task.elapsed():.1f}s)."
            ]
            if rendered.text:
                parts.append(rendered.text)
            parts.extend(
                [
                    self._output_notice(task),
                    f'- Poll:      shell_state(task_id="{task.task_id}")',
                    f'- Terminate: kill_shell(task_id="{task.task_id}")',
                ]
            )
            await self.runtime.agent_io.send(
                f"[{task.task_id}] promoted to background after {timeout}s"
            )
            return "\n".join(parts)

        rendered = self._render_output(
            task.head_lines, list(task.tail_lines), task.total_lines
        )
        parts = []
        if rendered.text:
            parts.append(rendered.text)
        if task.exit_code != 0:
            parts.append(f"[Process exited with code {task.exit_code}]")
        if rendered.truncated or not self._remove_output_file(task):
            parts.append(self._output_notice(task))
        self._tasks.pop(task.task_id, None)
        return "\n".join(parts)

    @tool()
    async def shell_state(self, task_id: str, description: str = "") -> str:
        """Check on a background (or promoted) task. Waits briefly (with
        per-task exponential backoff: 20s, 40s, 80s, ..., capped at 300s)
        so the model doesn't busy-poll, then returns new output, run status,
        total elapsed time, and the full captured-output path.

        New output uses the same 50-head/150-tail and 10 KiB output-data
        limits as foreground commands. The complete task output, including
        lines returned by earlier calls, remains in the session output file.

        The wait returns early if the process exits in the meantime. The
        backoff resets once the process reaches a terminal state.

        Args:
            task_id: The id returned by `shell(...)`.
            description: One short sentence on why you're checking
                (e.g., "Wait for dev server to bind port").
        """
        logger.info("Polling task %s (%s)", task_id, description)
        task = self._tasks.get(task_id)
        if task is None:
            return f"Unknown task_id: {task_id}"

        drain = task.drain_task
        if task.unread_lines == 0 and task.exit_code is None and drain is not None:
            delay = min(POLL_BASE_DELAY * (2**task.poll_count), POLL_MAX_DELAY)
            task.poll_count += 1
            task.new_output_event.clear()
            try:
                wait_event = asyncio.create_task(task.new_output_event.wait())
                _, pending = await asyncio.wait(
                    [asyncio.shield(drain), wait_event],
                    timeout=delay,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if wait_event in pending:
                    wait_event.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await wait_event
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

        head, tail, new_count = task.take_unread()
        if new_count > 0:
            task.poll_count = 0

        if task.exit_code is not None:
            status = f"exit {task.exit_code}"
            task.poll_count = 0
        else:
            status = "running"

        parts = [f"[{task_id}] status: {status} | elapsed: {task.elapsed():.1f}s"]
        if new_count:
            rendered = self._render_output(head, tail, new_count)
            if rendered.text:
                parts.append(rendered.text)
        else:
            parts.append("(no new output)")
        parts.append(self._output_notice(task))
        return "\n".join(parts)

    @tool()
    async def shell_input(self, task_id: str, text: str, description: str = "") -> str:
        """Send text to the stdin of a running background task.

        Args:
            task_id: The id of the running task.
            text: The text to send (e.g., "y\\n").
            description: Why you are sending this input.
        """
        logger.info("Sending input to task %s (%s): %r", task_id, description, text)
        task = self._tasks.get(task_id)
        if task is None:
            return f"Unknown task_id: {task_id}"
        if task.exit_code is not None:
            return f"Task {task_id} has already exited with code {task.exit_code}."

        if task.process.stdin is None:
            return f"Task {task_id} does not support stdin."

        try:
            if not text.endswith("\n"):
                logger.warning(
                    "shell_input received text without newline, appending \\n automatically."
                )
                text += "\n"
            task.process.stdin.write(text.encode())
            await task.process.stdin.drain()
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
        task = self._tasks.get(task_id)
        if task is None:
            return f"Unknown task_id: {task_id}"
        if task.process.returncode is not None:
            return (
                f"Task {task_id} already exited with code "
                f"{task.process.returncode} (elapsed {task.elapsed():.1f}s)"
            )
        await self._terminate(task.process)
        await self._flush_drain(task)
        logger.info(
            "Task %s killed: %s (exit %s, elapsed %.1fs)",
            task.task_id,
            task.description,
            task.process.returncode,
            task.elapsed(),
        )
        return (
            f"Killed task {task_id} "
            f"(exit code {task.process.returncode}, elapsed {task.elapsed():.1f}s)"
        )

    async def on_stop(self) -> None:
        drains = []
        for task_id in list(self._tasks):
            task = self._tasks.pop(task_id)
            if task.process.returncode is None:
                await self._terminate(task.process)
            if task.drain_task and not task.drain_task.done():
                task.drain_task.cancel()
                drains.append(task.drain_task)
        if drains:
            await asyncio.gather(*drains, return_exceptions=True)

    async def _get_info(self) -> str:
        if not self._tasks:
            return ""
        lines = ["", f"Background tasks ({len(self._tasks)}):"]
        for task_id, task in self._tasks.items():
            if task.exit_code is not None:
                state = f"exit {task.exit_code}"
            else:
                state = "running"
            lines.append(
                f"  - {task_id} [{state} {task.elapsed():.0f}s] {task.description}"
            )
        return "\n".join(lines)
