import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

from arox.agent_patterns.plugin import Plugin, tool
from arox.utils import truncate_content

logger = logging.getLogger(__name__)


def get_shell_context():
    import platform

    return {
        "os_info": platform.system(),
        "os_release": platform.release(),
        "shell_type": "bash",
    }


class ShellPlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)
        workspace_dir = self.agent.workspace.absolute()
        if not workspace_dir:
            raise ValueError("workspace_dir must be provided")

        self.workspace_dir = Path(workspace_dir)
        if not self.workspace_dir.is_absolute():
            raise ValueError(f"workspace_dir must be an absolute path: {workspace_dir}")

        if sys.platform == "linux":
            self.bwrap_path = shutil.which("bwrap")
            if not self.bwrap_path:
                raise RuntimeError("bwrap not found on linux, `Shell` tool disabled.")
        else:
            raise RuntimeError("No sandbox implemented. `Shell` tool disabled.")

    def _get_sandboxed_cmd(self, command: str) -> list[str]:
        """Construct the bwrap command arguments."""
        if sys.platform == "linux":
            return self._get_linux_sandboxed_cmd(command)
        else:
            return []

    def _get_linux_sandboxed_cmd(self, command: str) -> list[str]:
        workspace_str = str(self.workspace_dir)
        home_dir = Path.home()
        home_str = str(home_dir)

        bwrap_args = [
            self.bwrap_path,
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/sbin",
            "/sbin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            home_str,
            home_str,
            "--bind",
            workspace_str,
            workspace_str,
        ]

        # Mask sensitive directories/files in home
        sensitive_paths = [
            ".ssh",
            ".gnupg",
        ]
        for p in sensitive_paths:
            full_path = home_dir / p
            if full_path.exists():
                bwrap_args.extend(["--tmpfs", str(full_path)])

        bwrap_args.extend(
            [
                "--chdir",
                workspace_str,
                "--unshare-all",
                "--share-net",
                "--die-with-parent",
            ]
        )

        if os.path.exists("/lib64"):
            bwrap_args.extend(["--ro-bind", "/lib64", "/lib64"])

        # Essential files for networking and basic tools to work
        for path in [
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/passwd",
            "/etc/group",
            "/etc/ld.so.cache",
            "/etc/alternatives",
            "/etc/ssl",
            "/etc/ca-certificates",
        ]:
            if os.path.exists(path):
                bwrap_args.extend(["--ro-bind", path, path])

        bwrap_args.extend(["--", "/bin/bash", "-c", command])
        return bwrap_args

    @tool(dynamic_context=get_shell_context)
    async def shell(self, command: str, timeout: int | None = 100) -> str:
        """
        Run arbitrary shell commands in system's shell and return its output.

        Environment Info:
        - OS: {{ os_info }} {{ os_release }}
        - Shell: {{ shell_type }}

        Rules
            1. For searching code, use `rg` or `ast-grep`.
            2. Interactive commands that require user input are not supported and will fail.
            3. The command will be invoked by `bash -c`, mind the syntax. e.g.:
               - use single quote to avoid substution

        Examples
            command: "ls -la | rg staff"
            result: "total 24\\ndrwxr-xr-x  5 user  staff  160 Jan  1 12:00 .\\n..."

        Args:
            command: The shell command to execute (e.g., "ls -la", "pwd", "git status")
            timeout: Optional timeout in seconds for the command execution (default: 100)

        Returns:
            str: The combined stdout and stderr output of the command
        """
        try:
            logger.info(f"Executing shell command: {command}")
            sandboxed_cmd = self._get_sandboxed_cmd(command)

            env = os.environ.copy()

            process = await asyncio.create_subprocess_exec(
                *sandboxed_cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                error_msg = f"Command timed out after {timeout} seconds"
                logger.error(error_msg)
                return error_msg

            # Combine stdout and stderr
            output = stdout.decode()
            stderr_output = stderr.decode()
            if stderr_output:
                if output:
                    output += "\n"
                output += stderr_output

            # Truncate output if it's too large
            lines = output.splitlines()
            truncated = truncate_content(lines)
            output = "\n".join(truncated["lines"])
            if truncated["truncated_by_bytes"] or truncated["has_more_lines"]:
                output += f"\n\n[Output truncated due to size limits. Total lines: {len(lines)}]"

            # Add return code information
            if process.returncode != 0:
                output += f"\n[Process exited with code {process.returncode}]"

            logger.info(f"Command completed with return code: {process.returncode}")
            return output

        except Exception as e:
            error_msg = f"Error executing command: {e!s}"
            logger.error(error_msg)
            return error_msg
