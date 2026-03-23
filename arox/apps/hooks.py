import logging

from arox import commands

logger = logging.getLogger(__name__)


async def pre_llm_commit_hook(agent, input_content: str):
    logger.info("Running pre-LLM commit hook")


async def auto_compaction_hook(agent, input_content: str, result):
    if not result:
        return
    usage = result.usage()
    if usage and usage.request_tokens and usage.request_tokens > 100000:
        logger.info(
            f"Context size ({usage.request_tokens} tokens) exceeds threshold. Triggering automatic compaction."
        )
        await commands.CompactionCommand(agent).execute("compact", "")
