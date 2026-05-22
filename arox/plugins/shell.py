import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from arox.core.plugin import Plugin, tool
from arox.plugins.slots import AGENT_INFO, AGENT_RESET
from arox.utils import truncate_content

logger = logging.getLogger(__name__)

MAX_BACKGROUND_BUFFER_LINES = 5000
KILL_GRACE_SECONDS = 5
POLL_BASE_DELAY = 20
POLL_MAX_DELAY = 300


def get_shell_context():
    import os
    import platform
    import sys

    if sys.platform == "win32":
        shell_path = os.environ.get("COMSPEC", "cmd.exe")
    else:
        shell_path = os.environ.get("SHELL", "/bin/bash")
    shell_name = os.path.basename(shell_path)

    return {
        "os_info": platform.system(),
        "os_release": platform.release(),
        "shell_type": shell_name,
    }


@dataclass
class BackgroundShell:
    task_id: str
    command: str
    description: str
    process: asyncio.subprocess.Process
    started_at: float
    output_lines: deque = field(
        default_factory=lambda: deque(maxlen=MAX_BACKGROUND_BUFFER_LINES)
    )
    read_offset: int = 0
    exit_code: int | None = None
    finished_at: float | None = None
    drain_task: asyncio.Task | None = None
    poll_count: int = 0
    stream_to_io: bool = False
    notify_on_finish: bool = False

    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


class ShellPlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)
        self.workspace = self.agent.workspace.absolute()
        self._background: dict[str, BackgroundShell] = {}

        self.agent.provide_slot(AGENT_RESET, self._reset)
        self.agent.provide_slot(AGENT_INFO, self._get_info)

    def _get_cmd(self, command: str) -> list[str]:
        if sys.platform == "win32":
            shell_path = os.environ.get("COMSPEC", "cmd.exe")
            return [shell_path, "/c", command]
        else:
            shell_path = os.environ.get("SHELL", "/bin/bash")
            return [shell_path, "-c", command]

    async def _spawn(self, command: str) -> asyncio.subprocess.Process:
        cmd_args = self._get_cmd(command)
        kwargs: dict = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        return await asyncio.create_subprocess_exec(
            *cmd_args,
            cwd=str(self.workspace),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
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

    def _allocate_bg(
        self,
        process: asyncio.subprocess.Process,
        command: str,
        description: str,
        *,
        stream_to_io: bool,
        notify_on_finish: bool,
    ) -> BackgroundShell:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        bg = BackgroundShell(
            task_id=task_id,
            command=command,
            description=description,
            process=process,
            started_at=time.monotonic(),
            stream_to_io=stream_to_io,
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
                display = f"!{text}" if is_err else text
                bg.output_lines.append(display)
                if bg.stream_to_io:
                    try:
                        await self.agent.agent_io.send(f"$ {display}")
                    except Exception:
                        pass

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
        description: str,
        timeout: int | None = 100,
        run_in_background: bool = False,
    ) -> str:
        """Run arbitrary shell commands in the system's shell and return its output.

        Environment Info:
        - OS: {{ os_info }} {{ os_release }}
        - Shell: {{ shell_type }}

        Rules
            1. For searching code, use `rg` or `ast-grep`.
            2. Interactive commands that require user input are not supported and will fail.
            3. The command will be invoked by `{{ shell_type }} -c`, mind the syntax. e.g.:
               - use single quote to avoid substitution

        Long-running commands
            If a foreground command does not finish within `timeout` seconds,
            it is NOT killed — it is promoted to a background task and the
            returned message contains a `task_id` plus partial output. Poll
            it with `shell_state(task_id=...)` or kill it with
            `kill_shell(task_id=...)` if it runs too long.

            Set `run_in_background=True` explicitly when you already know the
            command is long-lived (dev server, watcher, multi-minute build).

        Examples
            command: "ls -la | rg staff"
            description: "List files matching 'staff'"
            result: "total 24\\ndrwxr-xr-x  5 user  staff  160 ..."

        Args:
            command: The shell command to execute.
            description: One short sentence describing what this command does
                (e.g., "Run unit tests"). Shown to the user and used in logs.
            timeout: Seconds to wait in the foreground before promoting to
                background (default 100). Ignored when `run_in_background=True`.
            run_in_background: If True, start detached and return a task id
                immediately.

        Returns:
            Foreground that finished in time: combined stdout/stderr (truncated
            if large) with an exit-code marker on non-zero exit.
            Foreground that timed out: promotion notice with task_id and
            partial output.
            Background: a task id and usage hint.
        """
        logger.info("Executing shell command (%s): %s", description, command)
        try:
            process = await self._spawn(command)
        except Exception as e:
            return f"Error spawning command: {e!s}"

        if run_in_background:
            bg = self._allocate_bg(
                process,
                command,
                description,
                stream_to_io=False,
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

        # Foreground: stream output live; promote to background on timeout.
        await self.agent.agent_io.send(f"$ {description}")
        bg = self._allocate_bg(
            process,
            command,
            description,
            stream_to_io=True,
            notify_on_finish=False,
        )
        drain = bg.drain_task
        assert drain is not None
        try:
            await asyncio.wait_for(asyncio.shield(drain), timeout=timeout)
        except TimeoutError:
            # Promote: keep process running, stop live-streaming, start
            # notifying on completion. Model can poll via shell_state.
            bg.stream_to_io = False
            bg.notify_on_finish = True
            tail_lines = list(bg.output_lines)[-50:]
            tail = "\n".join(tail_lines)
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

        # Completed in time: render and remove from registry.
        output = self._render_completed(bg)
        self._background.pop(bg.task_id, None)
        logger.info("Command completed with return code: %s", bg.exit_code)
        return output

    def _render_completed(self, bg: BackgroundShell) -> str:
        lines = list(bg.output_lines)
        truncated = truncate_content(lines)
        output = "\n".join(truncated["lines"])
        if truncated["truncated_by_bytes"] or truncated["has_more_lines"]:
            output += (
                f"\n\n[Output truncated due to size limits. Total lines: {len(lines)}]"
            )
        if bg.exit_code != 0:
            if output:
                output += "\n"
            output += f"[Process exited with code {bg.exit_code}]"
        return output

    @tool()
    async def shell_state(self, task_id: str, description: str) -> str:
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
        if bg.exit_code is None and drain is not None:
            delay = min(POLL_BASE_DELAY * (2**bg.poll_count), POLL_MAX_DELAY)
            bg.poll_count += 1
            try:
                await asyncio.wait_for(asyncio.shield(drain), timeout=delay)
            except TimeoutError:
                pass

        lines = list(bg.output_lines)
        new_lines = lines[bg.read_offset :]
        bg.read_offset = len(lines)

        if bg.exit_code is not None:
            status = f"exit {bg.exit_code}"
            bg.poll_count = 0
        else:
            status = "running"

        header = f"[{task_id}] status: {status} | elapsed: {bg.elapsed():.1f}s"

        if not new_lines:
            return f"{header}\n(no new output)"

        truncated = truncate_content(new_lines)
        body = "\n".join(truncated["lines"])
        if truncated["truncated_by_bytes"] or truncated["has_more_lines"]:
            body += f"\n[Output truncated. New lines this call: {len(new_lines)}]"
        return f"{header}\n{body}"

    @tool()
    async def kill_shell(self, task_id: str, description: str) -> str:
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
