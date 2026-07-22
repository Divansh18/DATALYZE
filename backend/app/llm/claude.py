"""Anthropic Claude client.

Single responsibility: talk to Claude.

This module does not load prompts, does not touch the database, does not
execute SQL, and does not know what MCP is. It receives a system prompt, a
message history and (optionally) tool definitions, sends them to Claude, and
returns the raw response for `chat/service.py` to orchestrate.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam

# Type aliases so the rest of the app doesn't import from `anthropic` directly.
Conversation = List[MessageParam]
ToolDefinition = ToolParam


class ClaudeClient:
    """Thin async wrapper around the Anthropic Messages API.

    Two entry points:

    * :meth:`complete` — one request, one full response. Use it for the
      tool-calling loop, where the orchestrator needs to inspect
      ``stop_reason`` and the ``tool_use`` blocks before continuing.
    * :meth:`stream` — same request, streamed. Use it for the final
      user-facing answer so tokens can be pushed over the WebSocket as they
      arrive.
    """

    DEFAULT_MODEL = "claude-opus-4-8"
    DEFAULT_MAX_TOKENS = 16000
    DEFAULT_EFFORT = "high"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        thinking: bool = True,
        timeout: float = 120.0,
    ) -> None:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Anthropic API key. Set ANTHROPIC_API_KEY in the environment."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(
        self,
        system: str,
        messages: Sequence[MessageParam],
        tools: Optional[Sequence[ToolDefinition]],
        max_tokens: Optional[int],
        cache_system: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": list(messages),
            "output_config": {"effort": self.effort},
        }

        if system:
            if cache_system:
                # The system prompt + DB schema are large and identical across
                # turns, so cache them and pay ~0.1x on every later request.
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system

        if self.thinking:
            payload["thinking"] = {"type": "adaptive"}

        if tools:
            payload["tools"] = list(tools)

        return payload

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def complete(
        self,
        system: str,
        messages: Sequence[MessageParam],
        tools: Optional[Sequence[ToolDefinition]] = None,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
    ) -> Message:
        """Send one request and return the complete response.

        The full ``Message`` is returned rather than a string because the
        orchestrator needs ``stop_reason`` and the ``tool_use`` blocks to drive
        the MCP loop. Append ``response.content`` verbatim to the conversation
        before sending tool results back — dropping thinking or tool_use blocks
        breaks the next turn.
        """
        request = self._build_request(system, messages, tools, max_tokens, cache_system)
        return await self._client.messages.create(**request)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def stream(
        self,
        system: str,
        messages: Sequence[MessageParam],
        tools: Optional[Sequence[ToolDefinition]] = None,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
    ):
        """Stream a response.

        Yields the SDK stream object. Iterate ``stream.text_stream`` for text
        deltas, then call ``await stream.get_final_message()`` for the
        assembled ``Message`` (needed to check ``stop_reason``)::

            async with claude.stream(system, messages) as s:
                async for chunk in s.text_stream:
                    await websocket.send_text(chunk)
                final = await s.get_final_message()
        """
        request = self._build_request(system, messages, tools, max_tokens, cache_system)
        async with self._client.messages.stream(**request) as stream:
            yield stream

    async def stream_text(
        self,
        system: str,
        messages: Sequence[MessageParam],
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
    ) -> AsyncIterator[str]:
        """Convenience wrapper: yield text chunks only.

        Use this for the final answer, once tool calling is finished.
        """
        async with self.stream(
            system, messages, max_tokens=max_tokens, cache_system=cache_system
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text(message: Message) -> str:
        """Concatenate the text blocks of a response."""
        return "".join(
            block.text for block in message.content if block.type == "text"
        ).strip()

    @staticmethod
    def wants_tools(message: Message) -> bool:
        """True when Claude is asking for a tool and the loop must continue."""
        return message.stop_reason == "tool_use"

    @staticmethod
    def extract_tool_uses(message: Message) -> List[Any]:
        """Return the tool_use blocks so the orchestrator can dispatch them.

        Each block carries ``.id``, ``.name`` and ``.input``. Every one of them
        must get a matching ``tool_result`` in the reply.
        """
        return [block for block in message.content if block.type == "tool_use"]

    @staticmethod
    def build_tool_results(results: Iterable[Dict[str, Any]]) -> MessageParam:
        """Wrap executed tool outputs into the user turn Claude expects.

        ``results`` is an iterable of
        ``{"tool_use_id": ..., "content": str, "is_error": bool}``.
        All results for one assistant turn go into a *single* user message —
        splitting them trains Claude to stop making parallel tool calls.
        """
        blocks: List[Dict[str, Any]] = []
        for result in results:
            block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": result["tool_use_id"],
                "content": result.get("content", ""),
            }
            if result.get("is_error"):
                block["is_error"] = True
            blocks.append(block)
        return {"role": "user", "content": blocks}

    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client (call on FastAPI shutdown)."""
        await self._client.close()


# ---------------------------------------------------------------------------
# Self-check:  python -m app.llm.claude               (run from backend/)
#              python -m app.llm.claude --live        (also calls the real API)
#
# The default run makes no network calls: it checks the request payload and the
# parsing helpers against synthetic responses. --live spends a few cents to
# confirm the key, the model id and the tool loop actually work end to end.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys

    from anthropic.types import Message as _Message

    from app.dev import check, report, section, skip

    LIVE = "--live" in sys.argv

    section("Configuration")

    with check("missing ANTHROPIC_API_KEY fails loudly at construction"):
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            try:
                ClaudeClient()
                raise AssertionError("should have raised ValueError")
            except ValueError as exc:
                assert "ANTHROPIC_API_KEY" in str(exc)
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY") or "sk-offline-test")

    section("Request payload")

    sample_tools = [
        {
            "name": "execute_sql",
            "description": "Run a SELECT.",
            "input_schema": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        }
    ]
    request = client._build_request(
        "SYSTEM", [{"role": "user", "content": "hi"}], sample_tools, None, True
    )

    with check("model id and max_tokens are set"):
        assert request["model"] == ClaudeClient.DEFAULT_MODEL
        assert request["max_tokens"] == ClaudeClient.DEFAULT_MAX_TOKENS

    with check("system is a top-level param, NOT a message role"):
        assert "system" in request
        roles = {m["role"] for m in request["messages"]}
        assert "system" not in roles, "system must never be a message role"

    with check("system prompt carries a cache_control breakpoint"):
        assert request["system"][0]["cache_control"] == {"type": "ephemeral"}

    with check("cache_system=False sends a plain string instead"):
        plain = client._build_request("SYSTEM", [], None, None, False)
        assert plain["system"] == "SYSTEM"

    with check("adaptive thinking is on; no removed budget_tokens field"):
        assert request["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in str(request)

    with check("no sampling params (they are rejected by this model)"):
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in request, f"{banned} would 400"

    with check("tools are passed through untouched"):
        assert request["tools"] == sample_tools

    with check("max_tokens override is honoured"):
        assert client._build_request("s", [], None, 512, True)["max_tokens"] == 512

    section("Response parsing")

    def _fake(content, stop_reason):
        return _Message.model_construct(
            id="msg_1",
            type="message",
            role="assistant",
            model=ClaudeClient.DEFAULT_MODEL,
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=None,
        )

    class _Blk:  # minimal stand-in for a content block
        def __init__(self, **kw):
            self.__dict__.update(kw)

    text_only = _fake([_Blk(type="text", text="Sales rose 12%.")], "end_turn")
    with_tool = _fake(
        [
            _Blk(type="thinking", thinking="need the data"),
            _Blk(type="text", text="Let me query that."),
            _Blk(
                type="tool_use",
                id="toolu_1",
                name="execute_sql",
                input={"sql": "SELECT 1"},
            ),
        ],
        "tool_use",
    )

    with check("extract_text concatenates only text blocks"):
        assert ClaudeClient.extract_text(text_only) == "Sales rose 12%."
        assert ClaudeClient.extract_text(with_tool) == "Let me query that."

    with check("wants_tools distinguishes a tool turn from a final answer"):
        assert ClaudeClient.wants_tools(with_tool) is True
        assert ClaudeClient.wants_tools(text_only) is False

    with check("extract_tool_uses skips thinking and text blocks"):
        uses = ClaudeClient.extract_tool_uses(with_tool)
        assert len(uses) == 1 and uses[0].name == "execute_sql"
        assert uses[0].input == {"sql": "SELECT 1"}
        assert ClaudeClient.extract_tool_uses(text_only) == []

    with check("all tool results land in ONE user message"):
        turn = ClaudeClient.build_tool_results(
            [
                {"tool_use_id": "toolu_1", "content": "3 rows"},
                {"tool_use_id": "toolu_2", "content": "boom", "is_error": True},
            ]
        )
        assert turn["role"] == "user"
        assert len(turn["content"]) == 2, "splitting these breaks parallel tool use"
        assert turn["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "3 rows",
        }
        assert turn["content"][1]["is_error"] is True

    with check("is_error is omitted on success, not set to False"):
        turn = ClaudeClient.build_tool_results([{"tool_use_id": "t", "content": "ok"}])
        assert "is_error" not in turn["content"][0]

    section("Live API call")

    if not LIVE:
        skip("real request to Anthropic", "pass --live to run (costs a few cents)")
    elif not os.getenv("ANTHROPIC_API_KEY"):
        skip("real request to Anthropic", "ANTHROPIC_API_KEY is not set")
    else:

        async def _live() -> None:
            live_client = ClaudeClient(max_tokens=1024)
            try:
                with check("complete() reaches the API and returns text"):
                    msg = await live_client.complete(
                        system="Answer with a single word.",
                        messages=[{"role": "user", "content": "Say: pong"}],
                        cache_system=False,
                    )
                    assert ClaudeClient.extract_text(msg), "empty response"
                    print(f"        -> {ClaudeClient.extract_text(msg)[:60]}")

                with check("Claude asks for the tool when data is needed"):
                    msg = await live_client.complete(
                        system="Use the execute_sql tool to answer questions about data.",
                        messages=[
                            {"role": "user", "content": "How many rows are in sales?"}
                        ],
                        tools=sample_tools,
                        cache_system=False,
                    )
                    assert ClaudeClient.wants_tools(msg), f"stop={msg.stop_reason}"
                    call = ClaudeClient.extract_tool_uses(msg)[0]
                    print(f"        -> {call.name}({call.input})")

                with check("stream_text() yields incremental chunks"):
                    chunks = [
                        c
                        async for c in live_client.stream_text(
                            system="Answer in one short sentence.",
                            messages=[{"role": "user", "content": "What is SQL?"}],
                            cache_system=False,
                        )
                    ]
                    assert chunks, "no chunks streamed"
                    print(f"        -> {len(chunks)} chunks, {sum(map(len, chunks))} chars")
            finally:
                await live_client.close()

        asyncio.run(_live())

    report("llm/claude.py" + (" [--live]" if LIVE else " [offline]"))
