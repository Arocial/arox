import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings


def deep_merge(source, overrides):
    """Deep merge two dictionaries, with overrides taking precedence"""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source:
            source[key] = deep_merge(source.get(key, {}), value)
        else:
            source[key] = value
    return source


def parse_dict(value: str) -> dict:
    """Parse a string of key=value pairs into a dictionary"""
    if not value.strip():
        return {}
    return dict(yaml.safe_load(value))


async def user_input_generator(completer=None, input=None, output=None):
    """Async generator that yields user input"""
    history = FileHistory(".arox_history")
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):  # Enter to submit
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter newline
    @kb.add("escape", "O", "M")  # Shift+Enter (at least in my konsole)
    def _(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(
        prompt_continuation="> ",
        multiline=True,
        key_bindings=kb,
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        mouse_support=False,
        completer=completer,
        input=input,
        output=output,
    )
    user_input = await session.prompt_async("\nUser (Ctrl+D to quit): ")
    return user_input


def xml_wrap(contents: list[tuple[str, str]]) -> str:
    xmled = []
    for tag, content in contents:
        if content is not None:
            xmled.append(f"<{tag}>\n{content}\n</{tag}>\n")
    return "\n".join(xmled)


async def run_command(command: str) -> tuple[str, str, int]:
    """
    Run a shell command asynchronously.

    Args:
        command: The command to run

    Returns:
        tuple: (stdout, stderr, return_code)
    """
    import asyncio

    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode
