import logging
import re
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic_ai import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    RunContext,
    UserPromptPart,
)

from arox.tools.file_edit import FileEdit

logger = logging.getLogger(__name__)


class ChatFiles:
    def __init__(self, agent, workspace, diff_agent) -> None:
        self.agent = agent
        self._chat_files = []
        self._pending_files = []
        self.candidate_generator = None
        self.workspace = workspace
        self.diff_agent = diff_agent
        self.file_tool = FileEdit(diff_agent)

    def normalize(self, path: str) -> Path:
        workspace = self.workspace
        # normalize file path to relative to workspace if it's subtree of workspace, otherwise absolute.
        p = Path(path)
        if not p.is_absolute():
            p = (workspace / p).absolute()
        if p.is_relative_to(workspace):
            p = p.relative_to(workspace)
        return p

    def add_by_names(self, paths: list[str]):
        succeed = []
        not_exist = []
        for path in paths:
            p = self.normalize(path)
            if not p.exists():
                not_exist.append(path)
                continue
            self.add(p)
            succeed.append(path)
        return {"succeed": succeed, "not_exist": not_exist}

    def add(self, f: Path):
        if f not in self._chat_files:
            self._chat_files.append(f)
        if f not in self._pending_files:
            self._pending_files.append(f)

    async def remove(self, f: Path):
        if f in self._chat_files:
            self._chat_files.remove(f)
        if f in self._pending_files:
            self._pending_files.remove(f)

    def clear(self):
        self._pending_files.clear()
        self._chat_files.clear()

    def have_pending(self):
        return bool(self._pending_files)

    def clear_pending(self):
        self._pending_files.clear()

    def list(self):
        return self._chat_files

    def set_candidate_generator(self, cg):
        self.candidate_generator = cg

    def candidates(self):
        if not self.candidate_generator:
            return []
        return self.candidate_generator()

    async def read_files(self):
        file_content = ""
        fpaths = []
        if not self._chat_files:
            return "", []

        # This is intended to check self._pending_files but add self._chat_files.
        for fname in self._chat_files:
            p = fname if fname.is_absolute() else self.workspace / fname
            try:
                with open(p, "r") as f:
                    content = f.read()
                    fpaths.append(fname)
                    logger.debug(f"Adding content from {fname}")
                    file_content = (
                        f"\n====FILE: {fname}====\n{content}\n\n{file_content}"
                    )
            except FileNotFoundError:
                await self.agent.io_channel.send(f"File not found: {p}")
                continue

        self.clear_pending()
        return file_content, fpaths

    async def update_contents(self, agent, input_content):
        """Parse LLM output for file edit operations and execute them.

        Args:
            output: LLM output containing xml tags like <replace_in_file> and <write_to_file>
        """
        # Access result through the state property
        if agent.state.result is None:
            return
        output = agent.state.result.output
        tags = ["replace_in_file", "write_to_file"]
        tags_re = f"{'|'.join(tags)}"
        pattern = rf'^<({tags_re})\s+path="([^"]+)">\n?(.*?)\n?^</\1>$'
        matches = re.findall(pattern, output, re.DOTALL | re.MULTILINE)
        for m in matches:
            method_str, path, content = m[0], m[1], m[2]
            method = getattr(self.file_tool, method_str)
            await method(path, content)


@dataclass(repr=False)
class UserFilesPart(UserPromptPart):
    user_part_kind: Literal["files"] = "files"

    def __str__(self) -> str:
        pattern = r"====FILE:\s*([^=]+)===="
        return "User provided files:\n" + "\n".join(
            [match.strip() for match in re.findall(pattern, str(self.content))]
        )


class SimpleState:
    def __init__(
        self,
        agent,
    ):
        self.agent = agent
        self.system_prompt = self.agent.system_prompt
        self.workspace = self.agent.workspace
        diff_agent = self.agent.context.get("diff_agent")
        self.chat_files = ChatFiles(agent, self.workspace, diff_agent)
        self.agent.add_after_step_hook(self.chat_files.update_contents)
        self._result = None
        self.reset()

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    @property
    def message_history(self):
        if self.result:
            return self.result.all_messages()
        else:
            return None

    async def assemble_chat_files(self) -> tuple[str, list[Path]]:
        return await self.chat_files.read_files()

    async def _update_parts(
        self,
        messages: list[ModelMessage],
        new_parts: list[UserPromptPart],
    ) -> list[ModelMessage]:
        remove_types = tuple([type(p) for p in new_parts])
        for msg in messages:
            if isinstance(msg, ModelRequest):
                parts = msg.parts
                msg.parts = [p for p in parts if not isinstance(p, remove_types)]

        if messages and isinstance(messages[-1], ModelRequest):
            last_request = messages[-1]
            parts = list(last_request.parts)
            parts.extend(new_parts)
            last_request.parts = parts

        return messages

    async def _parts_to_update(self) -> list[UserPromptPart]:
        if self.chat_files.have_pending():
            file_contents, _ = await self.assemble_chat_files()
            prompt_files = UserFilesPart(
                content=f"<files>\n{file_contents}\n</files>\n"
            )
            return [prompt_files]
        else:
            return []

    async def process_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        new_parts = await self._parts_to_update()
        messages = await self._update_parts(messages, new_parts)
        for msg in messages:
            for p in msg.parts:
                await self.agent.io_channel.send(f"{p}\n")
        return messages

    def reset(self):
        self.chat_files.clear()
        self.result = None

    async def handle_event(
        self, ctx: RunContext, events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.io_channel.send(event)
