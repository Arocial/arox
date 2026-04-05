import asyncio
import logging
import os
import sys

from arox.core.plugin import Plugin, tool
from arox.utils import truncate_content

logger = logging.getLogger(__name__)


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


class ShellPlugin(Plugin):
    def __init__(self, agent):
        super().__init__(agent)
        self.workspace = self.agent.workspace.absolute()

    def _get_cmd(self, command: str) -> list[str]:
        if sys.platform == "win32":
            shell_path = os.environ.get("COMSPEC", "cmd.exe")
            return [shell_path, "/c", command]
        else:
            shell_path = os.environ.get("SHELL", "/bin/bash")
            return [shell_path, "-c", command]

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
            3. The command will be invoked by `{{ shell_type }} -c`, mind the syntax. e.g.:
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
            cmd_args = self._get_cmd(command)

            env = os.environ.copy()

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                cwd=str(self.workspace),
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
