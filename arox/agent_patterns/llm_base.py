import json
import logging
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelSettings,
    mcp,
)

from arox import utils
from arox.agent_patterns.state import SimpleState
from arox.ui.io import AbstractIOAdapter, IOChannel

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    io_channel: AbstractIOAdapter


class LLMBaseAgent:
    def __init__(
        self,
        name,
        config_parser,
        io_adapter: AbstractIOAdapter,
        toolsets=[],
        state_cls=SimpleState,
        context={},
    ):
        self.uuid = str(uuid.uuid4())
        self.name = name
        self.context = context
        self.io_channel = IOChannel(adapter=io_adapter)
        self.model_ref = None
        self.additional_prompt = ""

        self.config_parser = config_parser
        self.config = self.parse_configs()
        # Manage tool specs.
        mcp_server_configs = self.config.mcp_servers or {}
        self.mcp_servers = []
        with tempfile.NamedTemporaryFile(mode="w") as tmpf:
            json.dump({"mcpServers": mcp_server_configs}, tmpf)
            tmpf.flush()
            self.mcp_servers = mcp.load_mcp_servers(tmpf.name)

        self.state = state_cls(self)

        self.pydantic_agent = Agent(
            self.provider_model,
            history_processors=[self.state.process_history],
            toolsets=toolsets + self.mcp_servers,
            deps_type=AgentDeps,
        )

    def set_model(self, model_ref: str):
        self.model_ref = model_ref
        config_parser = self.config_parser
        model_group = config_parser.add_argument_group(name=f"model.'{self.model_ref}'")
        model_group.add_argument("provider_model", default=self.model_ref)
        config_parser.add_argument_group(
            name=f"model.'{self.model_ref}'.params", expose_raw=True
        )
        config = config_parser.parse_args()
        model_config = getattr(config.model, self.model_ref)

        model_params = model_config.params
        self.model_params = utils.deep_merge(self.agent_model_params, model_params)
        self.provider_model = model_config.provider_model
        for model_prompt in self.model_aware_prompts:
            if re.search(model_prompt["pattern"], self.model_ref):
                self.additional_prompt = model_prompt["prompt"]

        return config

    async def show_agent_info(self):
        await self.io_channel.send(f"Using model {self.provider_model} for {self.name}")

    def parse_configs(self):
        config_parser = self.config_parser
        name = self.name
        agent_group = config_parser.add_argument_group(
            name=f"agent.{name}", expose_raw=True
        )
        agent_group.add_argument("system_prompt", default="")
        agent_group.add_argument("model_ref", default="")
        config_parser.add_argument_group(
            name=f"agent.{name}.model_params", expose_raw=True
        )
        config_parser.add_argument_group(
            name=f"agent.{name}.model_prompt", expose_raw=True
        )
        config = config_parser.parse_args()

        self.workspace = Path(config.workspace)
        if not self.workspace.is_absolute():
            self.workspace = self.workspace.absolute()
        group_config = getattr(config.agent, name)
        self.agent_config = group_config

        # Load default metadata using configargparse
        self.system_prompt = group_config.system_prompt
        self.model_ref = group_config.model_ref or config.model_ref
        self.agent_model_params = group_config.model_params
        self.model_aware_prompts = []
        mp = group_config.model_prompt
        for k, v in mp.items():
            if not k.endswith("_pattern"):
                pattern = mp.get(f"{k}_pattern", "")
                self.model_aware_prompts.append(
                    {
                        "prompt": v,
                        "pattern": pattern,
                    }
                )

        config = self.set_model(self.model_ref)
        return config

    async def _run_before_hooks(self, input_content: str):
        if hasattr(self, "before_step_hooks"):
            for hook in self.before_step_hooks:
                await hook(self, input_content)

    async def _run_after_hooks(self, input_content: str):
        if hasattr(self, "after_step_hooks"):
            for hook in self.after_step_hooks:
                await hook(self, input_content)

    async def step(self, input_content: str) -> AgentRunResult:
        await self._run_before_hooks(input_content)
        self.state.result = await self.pydantic_agent.run(
            input_content,
            event_stream_handler=self.state.handle_event,
            model_settings=ModelSettings(**self.model_params),
            instructions=f"{self.system_prompt}\n{self.additional_prompt}",
            message_history=self.state.message_history,
            deps=AgentDeps(io_channel=self.io_channel),
        )
        await self._run_after_hooks(input_content)
        return self.state.result

    def reset(self):
        return self.state.reset()

    def add_before_step_hook(self, hook):
        if not hasattr(self, "before_step_hooks"):
            self.before_step_hooks = []
        self.before_step_hooks.append(hook)

    def add_after_step_hook(self, hook):
        if not hasattr(self, "after_step_hooks"):
            self.after_step_hooks = []
        self.after_step_hooks.append(hook)
