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

logger = logging.getLogger(__name__)


class ChatFiles:
    def __init__(self, agent, workspace) -> None:
        self.agent = agent
        self._chat_files = []
        self._pending_files = []
        self.candidate_generator = None
        self.workspace = workspace

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
        self.chat_files = ChatFiles(agent, self.workspace)
        self.reset()

    async def assemble_chat_files(self) -> tuple[str, list[Path]]:
        return await self.chat_files.read_files()

    async def process_history(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        if self.chat_files.have_pending():
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    parts = msg.parts
                    msg.parts = [p for p in parts if not isinstance(p, UserFilesPart)]

            if messages and isinstance(messages[-1], ModelRequest):
                last_request = messages[-1]
                file_contents, _ = await self.assemble_chat_files()
                prompt_files = UserFilesPart(content=file_contents)
                parts = list(last_request.parts)
                parts.insert(0, prompt_files)
                last_request.parts = parts

        for msg in messages:
            for p in msg.parts:
                print(f"{p}\n")  # TODO send output event
        return messages

    def reset(self):
        self._messages = []
        self.message_meta = {}
        self.chat_files.clear()

    async def handle_event(
        self, ctx: RunContext, events: AsyncIterable[AgentStreamEvent]
    ):
        async for event in events:
            await ctx.deps.io_channel.send(event)
