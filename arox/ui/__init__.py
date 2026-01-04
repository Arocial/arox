import asyncio
import inspect
import logging
from enum import Enum
from typing import Callable, Iterable, Optional

from prompt_toolkit.completion import Completion
from prompt_toolkit.history import FileHistory, History
from pydantic_ai import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Collapsible, Footer, Label, ListItem, ListView, TextArea

from arox.ui.io import AbstractIOAdapter, EventNeedReply

logger = logging.getLogger(__name__)


def _hack_textual_keys():
    from textual._ansi_sequences import ANSI_SEQUENCES_KEYS

    class HKeys(str, Enum):
        @property
        def value(self) -> str:
            return super().value

        AltEnter = "alt+enter"

    ANSI_SEQUENCES_KEYS["\x1b\r"] = (HKeys.AltEnter,)


_hack_textual_keys()


class TUIByIO(App):
    BINDINGS = [
        ("f", "enable_follow", "Enable Follow Mode"),
        ("f", "disable_follow", "Disable Follow Mode"),
        ("c", "collapse_or_expand(True)", "Collapse All"),
        ("e", "collapse_or_expand(False)", "Expand All"),
    ]

    CSS = """
    Screen {
        padding: 1 1 0 1;
    }
    TextArea {
        width: 100%;
        height: auto;
        border: none;
    }
    .wrapped {
        width: 100%;
    }
    SuggestionPopup {
        background: $panel;
        border: round $accent;
        padding: 0;
        width: auto;
        height: auto;
        layer: popup;
        max-height: 10;
    }
    .input-container {
        layout: horizontal;
        width: 100%;
        height: auto;
        margin: 1 0;
    }
    .prompt-label {
        width: auto;
        color: $accent;
        padding: 0 1 0 0;
    }
    """

    def __init__(self, app_name):
        self.input_suggester = None
        self.app_name = app_name
        self.io_adapter = TextualIOAdapter(self, title=app_name)
        self.follow = True
        super().__init__()

    def on_mount(self) -> None:
        self.theme = "nord"

    def compose(self) -> ComposeResult:
        yield Footer()

    def action_collapse_or_expand(self, collapse: bool) -> None:
        for child in self.walk_children(Collapsible):
            child.collapsed = collapse

    def action_disable_follow(self):
        self.follow = False
        self.refresh_bindings()

    def action_enable_follow(self):
        self.follow = True
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "enable_follow" and self.follow:
            return False
        elif action == "disable_follow" and not self.follow:
            return False
        return True

    async def update_and_scroll(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result

        if self.follow:
            self.call_later(self.screen.scroll_end)


class SuggestionPopup(ListView):
    """A popup widget for displaying suggestions."""

    def __init__(self, suggestions: list[Completion], *args, **kwargs):
        self.suggestions = suggestions
        items = [ListItem(Label(suggestion.display_text)) for suggestion in suggestions]
        super().__init__(*items, *args, **kwargs)

    def get_selected_suggestion(self) -> Completion:
        if 0 <= self.index < len(self.suggestions):
            return self.suggestions[self.index]
        return Completion("")

    def move_selection(self, direction: int):
        """Move selection up (-1) or down (1)."""
        new_index = self.index + direction
        if 0 <= new_index < len(self.suggestions):
            self.index = new_index


class UserInput(TextArea):
    BINDINGS = [
        ("ctrl+j", "submit", "Submit"),
        ("alt+enter", "submit", "Submit"),
        ("up", "on_up", "Previous"),
        ("down", "on_down", "Next"),
        ("ctrl+r", "history_search", "Search History"),
        ("ctrl+g", "on_abort", "Abort"),
        ("escape", "on_abort", "Abort"),
        ("tab", "on_down", "Accept Suggestion"),
    ]

    def __init__(
        self,
        input_future,
        history: History = None,
        suggester: Optional[Callable[[TextArea], Iterable[Completion]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_future = input_future
        self.submit_history = history
        self.history_index = -1  # -1 means no history active
        self.history_search_text = ""
        self.history_search_mode = False
        self.suggester = suggester
        self.suggestion_popup: Optional[SuggestionPopup] = None
        self.suggestions_visible = False

    async def _on_key(self, event: events.Key) -> None:
        continue_super = True

        if self.history_search_mode:
            continue_super = False
            if not event.is_printable:
                await self.history_search_exit()
            else:
                event.prevent_default()
                self.history_search_text += event.character
                await self.action_history_search()

        elif self.suggestions_visible:
            if event.key == "enter":
                continue_super = False
                event.prevent_default()
                await self.action_accept_suggestion()
            elif event.key not in ("up", "down", "tab"):
                await self.action_hide_suggestions()

        # Textual parses shift+enter this way.
        # One can use [tkrec](https://github.com/Textualize/textual-key-recorder/)
        # to see the keys result in textual app.
        if event.key == "enter" and event.character is None:
            event.prevent_default()
            await self.action_submit()
            continue_super = False

        if continue_super:
            await super()._on_key(event)
            if event.is_printable and self.suggester:
                await self.update_suggestions()

    async def action_submit(self) -> None:
        """Submit the input."""
        user_input = self.text
        await self.action_hide_suggestions()
        if self.submit_history and user_input:
            self.submit_history.append_string(user_input)
        if self.input_future and not self.input_future.done():
            self.input_future.set_result(user_input)

    @property
    async def loaded_history(self):
        return [h async for h in self.submit_history.load()]

    async def action_history_search(self):
        """Search history matching current input."""
        if not self.submit_history:
            return

        if not self.history_search_mode:
            self.history_search_mode = True
            self.history_search_text = self.text
        history = await self.loaded_history
        for i in range(self.history_index, len(history)):
            if self.history_search_text.lower() in history[i].lower():
                self.history_index = i
                self.text = (
                    f"(search-history) `{self.history_search_text}`: {history[i]}"
                )
                break
        else:
            self.text = f"(search-history) `{self.history_search_text}`: {history[self.history_index]}"

    async def history_search_exit(self, find=True):
        if self.history_search_mode:
            self.text = (
                (await self.loaded_history)[self.history_index]
                if find
                else self.history_search_text
            )
            self.history_index = -1
            self.history_search_mode = False

    async def action_on_abort(self):
        """Reset history navigation state."""
        if self.suggestions_visible and self.suggestion_popup:
            await self.action_hide_suggestions()
            return

        if self.history_search_mode:
            await self.history_search_exit(find=False)
            self.text = self.history_search_text

    async def action_on_up(self):
        """Navigate to previous suggestion item or history item."""
        if self.suggestions_visible and self.suggestion_popup:
            self.suggestion_popup.move_selection(-1)
            return

        if not self.submit_history:
            return
        await self.history_search_exit()
        history = await self.loaded_history
        if self.history_index < len(history) - 1:
            self.history_index += 1
            self.text = history[self.history_index]

    async def action_on_down(self):
        """Navigate to next suggestion item or history item."""
        if self.suggestions_visible and self.suggestion_popup:
            self.suggestion_popup.move_selection(1)
            return

        if not self.submit_history:
            return
        await self.history_search_exit()
        history = await self.loaded_history
        if self.history_index > 0:
            self.history_index -= 1
            self.text = history[self.history_index]
        elif self.history_index == 0:
            self.history_index = -1
            self.text = self.history_search_text

    async def update_suggestions(self):
        """Update suggestions based on current text."""
        if not self.suggester:
            return

        suggestions = list(self.suggester(self))

        if suggestions:
            await self.show_suggestions(suggestions)
        else:
            await self.action_hide_suggestions()

    async def show_suggestions(self, suggestions: list[Completion]):
        """Show the suggestion popup with given suggestions."""
        await self.action_hide_suggestions()  # Hide any existing popup

        if not suggestions:
            return

        # Too many suggestions slows the ui significantly.
        self.suggestion_popup = SuggestionPopup(suggestions[:20])

        # Position popup intelligently
        cursor_offset = self.cursor_screen_offset
        # Get the size of the terminal
        terminal_size = self.app.size
        popup_width = (
            max(len(s.display_text) for s in suggestions) + 6
        )  # Add some padding
        popup_height = min(len(suggestions), 10) + 2  # Limit height to 10 items

        # Calculate x position (prevent right overflow)
        matched_len = suggestions[0].start_position if suggestions else 0
        x_pos = min(cursor_offset[0] + matched_len, terminal_size.width - popup_width)

        # Calculate y position (flip to above if below is not enough space)
        if cursor_offset[1] + popup_height <= terminal_size.height:
            # Place below cursor
            y_pos = cursor_offset[1]
        else:
            # Place above cursor
            y_pos = cursor_offset[1] - popup_height

        popup_offset = (x_pos, y_pos)
        self.suggestion_popup.styles.offset = popup_offset
        self.suggestion_popup.styles.width = popup_width

        await self.app.mount(self.suggestion_popup)
        self.suggestions_visible = True

    async def action_hide_suggestions(self):
        """Hide the suggestion popup."""
        if self.suggestion_popup:
            await self.suggestion_popup.remove()
            self.suggestion_popup = None
        self.suggestions_visible = False

    async def action_accept_suggestion(self):
        """Accept the selected suggestion."""
        if self.suggestion_popup:
            suggestion = self.suggestion_popup.get_selected_suggestion()
            if suggestion:
                cl = self.cursor_location
                # Assume the matched part don't across lines.
                sl = (cl[0], max(0, cl[1] + suggestion.start_position))
                self.replace(suggestion.text, sl, cl)
        await self.action_hide_suggestions()


class CollapsibleLabel(Collapsible):
    def __init__(self, content="", title="", collapsed=True):
        self.label_widget = Label(content, markup=False, classes="wrapped")
        super().__init__(self.label_widget, title=title, collapsed=collapsed)

    def update(self, content):
        self.label_widget.update(content)


class TextualIOAdapter(AbstractIOAdapter):
    def __init__(self, app, title=""):
        self.app = app
        self.output_widget = None
        self.title = title
        self.current_stream_widget = None

    async def _handle_output(self, event):
        if isinstance(event, PartStartEvent):
            part = event.part
            self.current_stream_widget = StreamingOutputWidget(self.app, part.part_kind)
            await self.current_stream_widget.start()
            if isinstance(part, (TextPart, ThinkingPart)):
                await self.current_stream_widget.write(f"{part.content}")
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, (TextPartDelta, ThinkingPartDelta)):
                await self.current_stream_widget.write(event.delta.content_delta)
            elif isinstance(event.delta, ToolCallPartDelta):
                await self.current_stream_widget.write(event.delta.args_delta)
        elif isinstance(event, PartEndEvent):
            await self.current_stream_widget.end()
        elif isinstance(event, FunctionToolResultEvent):
            await self.write(
                f"<tool_result> {event.tool_call_id!r} returned => {event.result.content}</tool_result>\n"
            )
        elif isinstance(event, (FunctionToolCallEvent, FinalResultEvent)):
            pass
        elif isinstance(event, EventNeedReply):
            reply = await self.read()
            event.set_reply(reply)
        else:
            await self.write(f"\nUnexpected event type: {event.__class__.__name__}\n")

    async def handle_output_stream(self, output_stream_out):
        async with output_stream_out:
            async for event in output_stream_out:
                await self._handle_output(event)

    async def read(self):
        input_future = asyncio.get_running_loop().create_future()
        from textual.containers import Container
        from textual.widgets import Static

        prompt_label = Static(f"{self.title} > ", classes="prompt-label")

        input_widget = UserInput(
            input_future,
            history=FileHistory(f".arox.{self.title}.history"),
            suggester=self.app.input_suggester,
        )

        input_container = Container(
            prompt_label, input_widget, classes="input-container"
        )

        await self.app.mount(input_container)
        input_widget.focus()
        user_input = await input_future
        await input_container.remove()

        collapsible = Collapsible(
            Label(user_input, markup=False, classes="wrapped"),
            collapsed=False,
            title=f"{self.title}.User",
        )
        await self.app.update_and_scroll(self.app.mount, collapsible)
        return user_input

    async def write(self, content, metadata=None):
        output_content = f"{self.title}: {str(content)}"
        output_widget = Label(output_content)
        await self.app.update_and_scroll(self.app.mount, output_widget)


def _process_xml_tags(content_str, title, app):
    """Process content with XML tags and return appropriate widget."""
    import re

    # Regular expression to match XML-style tags
    xml_tag_pattern = r"^<(\w+)([^>]*)>(.*?)</\1>"

    # Find all XML tags and their content
    matches = list(re.finditer(xml_tag_pattern, content_str, re.DOTALL | re.MULTILINE))

    if matches:
        # Process content with XML tags
        sub_widgets = []

        # Add content before the first tag if it exists
        last_end = 0
        for match in matches:
            # Add content before this tag if it's not empty
            pre_content = content_str[last_end : match.start()].strip()
            if pre_content:
                output_widget = CollapsibleLabel(
                    pre_content,
                    title=f"{title}.text",
                    collapsed=True,
                )
                sub_widgets.append(output_widget)

            # Extract tag name and content
            tag_name = match.group(1)
            tag_attrs = match.group(2).strip()
            tag_content = match.group(3).strip()

            # Create a title for the collapsible section
            widget_title = f"{title}.{tag_name}"
            if tag_attrs:
                widget_title += f" {tag_attrs}"

            output_widget = CollapsibleLabel(
                tag_content,
                title=widget_title,
                collapsed=True,
            )
            sub_widgets.append(output_widget)

            last_end = match.end()

        # Add remaining content after the last tag if it exists
        remaining_content = content_str[last_end:].strip()
        if remaining_content:
            output_widget = CollapsibleLabel(
                remaining_content,
                title=f"{title}.text",
                collapsed=True,
            )
            sub_widgets.append(output_widget)

        main_collapsible = Collapsible(
            *sub_widgets,
            title=title,
            collapsed=False,
        )

        return main_collapsible
    else:
        # If no XML tags found, return a simple CollapsibleLabel
        return CollapsibleLabel(
            content_str,
            title=title,
            collapsed=True,
        )


class PromptMessageWidget:
    def __init__(self, app, title=""):
        self.app = app
        self.title = title

    async def write(self, content, metadata=None):
        # Ensure content is a string
        content_str = (
            "\n".join([c.get("content", "") for c in content])
            if isinstance(content, list)
            else str(content)
        )

        # Process content with XML tags and get appropriate widget
        widget = _process_xml_tags(content_str, self.title, self.app)

        # Mount the widget
        await self.app.update_and_scroll(self.app.mount, widget)
        widget.focus()


class StreamingOutputWidget:
    def __init__(self, app, title=""):
        self.app = app
        self.output_widget = None
        self.title = title
        self.accumulated_content = ""

    async def start(self):
        if self.output_widget is None:
            self.output_widget = CollapsibleLabel(title=self.title, collapsed=False)
            await self.app.update_and_scroll(self.app.mount, self.output_widget)
            self.output_widget.focus()

    async def write(self, content, metadata=None):
        if content:
            self.accumulated_content += str(content)
            await self.app.update_and_scroll(
                self.output_widget.update, self.accumulated_content
            )

    async def end(self):
        # Process content with XML tags and get appropriate widget
        widget = _process_xml_tags(self.accumulated_content, self.title, self.app)

        # If we already have an output widget, remove it before adding the new one
        if self.output_widget:
            await self.output_widget.remove()

        # Update the output widget reference and mount
        self.output_widget = widget
        await self.app.update_and_scroll(self.app.mount, self.output_widget)
        self.output_widget.focus()
