"""Unified completion protocol shared by all IO adapters.

Two pieces:

* :class:`CompletionRequest` / :class:`CompletionItem` — UI-agnostic data
  shape. UI adapters build a request from whatever local input model they
  have (a prompt-toolkit ``Document`` or a raw HTTP query) and translate
  the returned items back into their native completion type.
* :class:`CompletionRouter` — registry that fans the request out to the
  right provider based on trigger character (``/``, ``@``, ...) and
  current token. Slash sub-completers and non-slash mention providers
  share a single dispatch path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

CompletionProvider = Callable[["CompletionRequest"], Iterable["CompletionItem"]]

DEFAULT_RESULT_LIMIT = 50


@dataclass
class CompletionItem:
    """A single completion candidate.

    ``replace_range`` is in absolute positions over ``CompletionRequest.text``.
    If unset, the router fills it with the bounds of ``current_token``.
    """

    value: str
    label: str | None = None
    description: str | None = None
    group: str | None = None
    score: float = 0.0
    replace_range: tuple[int, int] | None = None


@dataclass
class CompletionRequest:
    text: str
    cursor: int
    trigger: str | None = None
    tokens: list[str] = field(default_factory=list)
    current_token: str = ""
    current_token_range: tuple[int, int] = (0, 0)
    agent: Any | None = None


def parse_request(
    text: str,
    cursor: int | None = None,
    *,
    triggers: tuple[str, ...] = ("/", "@"),
    agent: Any | None = None,
) -> CompletionRequest:
    """Build a :class:`CompletionRequest` from raw input text.

    Splits on whitespace, computes the token under ``cursor`` and detects
    the leading trigger character if any. Behaviour stays predictable when
    ``cursor`` falls on a trailing space — the current token is empty and
    its range is a zero-width span at the cursor.
    """
    if cursor is None:
        cursor = len(text)
    cursor = max(0, min(cursor, len(text)))

    trigger = text[0] if text and text[0] in triggers else None

    # find token under cursor
    start = cursor
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = cursor
    while end < len(text) and not text[end].isspace():
        end += 1
    current_token = text[start:end]

    tokens = text.split()
    return CompletionRequest(
        text=text,
        cursor=cursor,
        trigger=trigger,
        tokens=tokens,
        current_token=current_token,
        current_token_range=(start, end),
        agent=agent,
    )


class CompletionRouter:
    """Aggregates slash-command and trigger-based completion providers."""

    def __init__(self, *, limit: int = DEFAULT_RESULT_LIMIT):
        self._limit = limit
        # /<name> top-level menu items, populated as slash commands register
        self._slash_top: dict[str, CompletionItem] = {}
        # /<name> ... -> per-name sub-provider
        self._slash_sub: dict[str, CompletionProvider] = {}
        # '@' / '#' / ... -> ordered providers
        self._triggers: dict[str, list[CompletionProvider]] = {}

    def register_slash(
        self,
        name: str,
        *,
        description: str = "",
        sub: CompletionProvider | None = None,
    ) -> None:
        self._slash_top[name] = CompletionItem(
            value=f"/{name}",
            label=f"/{name}",
            description=description or None,
            group="command",
        )
        if sub is not None:
            self._slash_sub[name] = sub

    def register_trigger(self, char: str, provider: CompletionProvider) -> None:
        self._triggers.setdefault(char, []).append(provider)

    @property
    def slash_top(self) -> dict[str, CompletionItem]:
        return self._slash_top

    def complete(self, req: CompletionRequest) -> list[CompletionItem]:
        items: list[CompletionItem] = []
        if req.trigger == "/":
            items.extend(self._complete_slash(req))
        elif req.trigger and req.trigger in self._triggers:
            for provider in self._triggers[req.trigger]:
                items.extend(provider(req))

        # Fill in replace_range default and clamp
        for it in items:
            if it.replace_range is None:
                it.replace_range = req.current_token_range
        return items[: self._limit]

    def _complete_slash(self, req: CompletionRequest) -> Iterable[CompletionItem]:
        # Determine whether we're completing the command name or its arguments.
        # Treat "/foo" (no trailing space) as still editing the name; "/foo "
        # (with separator after the name) as editing arguments.
        text = req.text
        first_space = text.find(" ")
        editing_name = first_space == -1 or req.cursor <= first_space

        if editing_name:
            typed = (
                req.tokens[0][1:]
                if req.tokens and req.tokens[0].startswith("/")
                else ""
            )
            typed_lower = typed.lower()
            name_range = (0, first_space if first_space != -1 else len(text))
            for name, item in self._slash_top.items():
                if typed_lower and typed_lower not in name.lower():
                    continue
                yield CompletionItem(
                    value=item.value,
                    label=item.label,
                    description=item.description,
                    group=item.group,
                    score=_score(typed_lower, name.lower()),
                    replace_range=name_range,
                )
            return

        name = req.tokens[0][1:]
        sub = self._slash_sub.get(name)
        if sub is None:
            return
        yield from sub(req)


def _score(query: str, candidate: str) -> float:
    if not query:
        return 0.0
    if candidate.startswith(query):
        return 2.0
    if query in candidate:
        return 1.0
    return 0.0
