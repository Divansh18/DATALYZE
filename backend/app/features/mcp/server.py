"""MCP server entry point.

Single responsibility: initialize the MCP server, register the tools defined in
`tools.py`, and start it.

No SQL, no Claude, no business logic. Every tool body here is a one-line call
into `tools.dispatch`, so the server and the in-process path used by
`chat/service.py` can never drift apart.

Run standalone (stdio transport):

    python -m app.features.mcp.server
"""

import logging

from mcp.server.fastmcp import FastMCP

from . import tools

logger = logging.getLogger(__name__)

mcp = FastMCP("datalyze-db")


@mcp.tool(
    name="get_schema",
    description=(
        "Return the database schema: tables, columns, data types and "
        "relationships. Call this before writing SQL for a question you have "
        "not queried yet — never guess table or column names."
    ),
)
def get_schema() -> str:
    """Describe the PostgreSQL database."""
    return tools.dispatch("get_schema")


@mcp.tool(
    name="execute_sql",
    description=(
        "Run a read-only SQL SELECT statement against PostgreSQL and return the "
        "rows. Only a single SELECT is permitted; INSERT, UPDATE, DELETE, DROP "
        "and multi-statement input are rejected. If the call fails, read the "
        "message, correct the SQL and try again."
    ),
)
def execute_sql(sql: str) -> str:
    """Execute one validated SELECT statement.

    Args:
        sql: A single PostgreSQL SELECT statement. Include an explicit LIMIT
            unless the query is already aggregated.
    """
    return tools.dispatch("execute_sql", {"sql": sql})


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logger.info("Starting MCP server with tools: %s", ", ".join(tools.tool_names()))
    mcp.run(transport="stdio")


# ---------------------------------------------------------------------------
# Self-check:  python -m app.features.mcp.server --selftest    (from backend/)
#
# Verifies the server registers the tools and that each one is reachable,
# without opening a stdio transport (which would block waiting for a client).
# Run with no arguments to actually start the server.
# ---------------------------------------------------------------------------


def _selftest() -> None:
    import asyncio
    import json

    from app.dev import check, report, section

    # FastMCP installs its own rich logging handler at import time, so
    # basicConfig() will not silence the tracebacks this test provokes on
    # purpose. disable() suppresses them regardless of handler.
    logging.disable(logging.CRITICAL)

    # Stub the database layer so the tools can run without PostgreSQL.
    from app.features.database import executor, schema, validator

    validator.validate_sql = lambda _sql: None                       # type: ignore[assignment]
    executor.execute_query = lambda _sql: [{"ok": 1}]                # type: ignore[assignment]
    schema.get_database_schema = lambda: {"sales": {"columns": {}}}  # type: ignore[assignment]

    section("Registration")

    registered = asyncio.run(mcp.list_tools())
    names = sorted(t.name for t in registered)

    with check("server registers exactly the tools in the registry"):
        assert names == tools.tool_names(), f"{names} != {tools.tool_names()}"

    with check("every registered tool carries a description for Claude"):
        for tool in registered:
            assert (tool.description or "").strip(), f"{tool.name} has no description"

    with check("execute_sql declares a required 'sql' string parameter"):
        spec = next(t for t in registered if t.name == "execute_sql")
        params = spec.inputSchema
        assert params["properties"]["sql"]["type"] == "string", params
        assert params.get("required") == ["sql"], params

    section("Invocation through the MCP layer")

    with check("get_schema returns a successful payload"):
        result = asyncio.run(mcp.call_tool("get_schema", {}))
        assert json.loads(_text_of(result))["ok"] is True

    with check("execute_sql returns rows through the MCP layer"):
        result = asyncio.run(mcp.call_tool("execute_sql", {"sql": "SELECT 1"}))
        payload = json.loads(_text_of(result))
        assert payload["ok"] is True and payload["row_count"] == 1, payload

    with check("a tool failure surfaces as text, not a transport error"):
        executor.execute_query = _raise_conn  # type: ignore[assignment]
        result = asyncio.run(mcp.call_tool("execute_sql", {"sql": "SELECT 1"}))
        payload = json.loads(_text_of(result))
        assert payload["ok"] is False and "refused" in payload["error"], payload

    report("mcp/server.py")


def _raise_conn(_sql: str):
    raise ConnectionError("connection refused")


def _text_of(result) -> str:
    """Pull the text payload out of whatever shape call_tool returned.

    FastMCP has returned a bare content list in some versions and a
    (content, structured) tuple in others; handle both.
    """
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        return result[0].text
    return str(result)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
