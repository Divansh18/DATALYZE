"""Tool definitions exposed to Claude.

Single responsibility: describe each tool and wire it to the right
database-layer function.

No business logic, no SQL, no connection handling — validation lives in
`database/validator.py`, execution in `database/executor.py`, introspection in
`database/schema.py`, and result formatting in `mcp/service.py`.
"""

import logging
from typing import Any, Callable, Dict, List

from app.features.database import executor, schema, validator

from . import service
from .service import ToolExecutionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
#
# One source of truth, in Anthropic tool format. `mcp/server.py` re-registers
# the same callables with the MCP server, so a tool is described once here and
# nowhere else.
#
# Descriptions are prescriptive about *when* to call each tool, not just what
# it does — that measurably raises how reliably Claude reaches for them.
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "get_schema",
        "description": (
            "Return the database schema: tables, columns, data types and "
            "relationships. Call this before writing any SQL for a question you "
            "have not queried yet in this conversation — never guess table or "
            "column names. Safe to call once per conversation; the result does "
            "not change between turns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "execute_sql",
        "description": (
            "Run a read-only SQL SELECT statement against the PostgreSQL "
            "database and return the rows. Call this whenever answering the "
            "user requires actual data. Only a single SELECT statement is "
            "permitted — INSERT, UPDATE, DELETE, DROP and multi-statement input "
            "are rejected. If the statement is rejected or errors, read the "
            "message, correct the SQL and call the tool again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A single PostgreSQL SELECT statement. Include an "
                        "explicit LIMIT unless the query is already aggregated."
                    ),
                }
            },
            "required": ["sql"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
#
# Each returns a plain dict; `service.run_tool` serializes it and converts any
# exception into readable text for Claude.
# ---------------------------------------------------------------------------


def _get_schema() -> Dict[str, Any]:
    return service.describe_schema(schema.get_database_schema())


def _execute_sql(sql: str) -> Dict[str, Any]:
    if not sql or not sql.strip():
        raise ToolExecutionError("No SQL was provided.")

    # Safety gate first — the executor must never see unvalidated SQL.
    try:
        validator.validate_sql(sql)
    except Exception as exc:  # validator raises its own error type
        raise ToolExecutionError(
            f"This statement was rejected: {exc} "
            "Only a single read-only SELECT statement is allowed."
        ) from exc

    rows = executor.execute_query(sql)
    result = service.format_rows(rows)
    result["sql"] = sql.strip()
    return result


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_schema": _get_schema,
    "execute_sql": _execute_sql,
}


def dispatch(name: str, arguments: Dict[str, Any] | None = None) -> str:
    """Run the named tool and return its serialized result.

    This is the entry point used both by `mcp/server.py` and by
    `chat/service.py` when it resolves a `tool_use` block from Claude. The
    return value is always a string and never raises — a failed tool comes back
    as error text so the conversation can continue.
    """
    fn = _REGISTRY.get(name)
    if fn is None:
        known = ", ".join(sorted(_REGISTRY))
        return service.failure(
            f"Unknown tool {name!r}.", hint=f"Available tools: {known}."
        )

    logger.info("Dispatching tool %s", name)
    return service.run_tool(name, fn, **(arguments or {}))


def tool_names() -> List[str]:
    return sorted(_REGISTRY)


__all__ = ["TOOL_DEFINITIONS", "dispatch", "tool_names"]


# ---------------------------------------------------------------------------
# Self-check:  python -m app.features.mcp.tools        (run from backend/)
#
# The database layer is stubbed, so this runs with no PostgreSQL and no
# credentials. It verifies the wiring: schemas are well-formed, the validator
# runs before the executor, and every failure path returns text instead of
# raising. Add --real to run against the actual database layer instead.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    from app.dev import check, report, section

    logging.basicConfig(level=logging.CRITICAL)  # silence expected tracebacks

    USE_REAL_DB = "--real" in sys.argv

    if not USE_REAL_DB:
        # Replace the database layer with fakes. `tools.py` must not care where
        # rows come from, so this is a fair test of everything except the DB.
        calls: List[str] = []

        class _Rejected(Exception):
            pass

        def _fake_validate(sql: str) -> None:
            calls.append("validate")
            lowered = sql.strip().lower()
            if not lowered.startswith("select"):
                raise _Rejected("statement is not a SELECT")

        def _fake_execute(sql: str) -> List[Dict[str, Any]]:
            calls.append("execute")
            return [
                {"product": "Widget", "revenue": 1250.0},
                {"product": "Gadget", "revenue": 980.0},
            ]

        def _fake_schema() -> Dict[str, Any]:
            calls.append("schema")
            return {
                "sales": {
                    "columns": {"id": "integer", "product": "text", "revenue": "numeric"}
                }
            }

        validator.validate_sql = _fake_validate       # type: ignore[assignment]
        executor.execute_query = _fake_execute        # type: ignore[assignment]
        schema.get_database_schema = _fake_schema     # type: ignore[assignment]

    section("Tool schemas (what Claude actually receives)")

    with check("every definition has name/description/input_schema"):
        for tool in TOOL_DEFINITIONS:
            assert set(tool) == {"name", "description", "input_schema"}, tool
            assert tool["description"].strip(), tool["name"]

    with check("schema names match the dispatch registry exactly"):
        declared = sorted(t["name"] for t in TOOL_DEFINITIONS)
        assert declared == tool_names(), f"{declared} != {tool_names()}"

    with check("execute_sql requires a 'sql' string argument"):
        spec = next(t for t in TOOL_DEFINITIONS if t["name"] == "execute_sql")
        props = spec["input_schema"]["properties"]
        assert spec["input_schema"]["required"] == ["sql"]
        assert props["sql"]["type"] == "string"

    with check("get_schema takes no arguments"):
        spec = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_schema")
        assert spec["input_schema"]["properties"] == {}
        assert spec["input_schema"]["required"] == []

    section("Dispatch")

    with check("dispatch always returns a JSON string, never raises"):
        out = dispatch("get_schema")
        assert isinstance(out, str)
        assert json.loads(out)["ok"] is True

    with check("unknown tool name is reported, not raised"):
        out = json.loads(dispatch("drop_everything"))
        assert out["ok"] is False and "Unknown tool" in out["error"]
        assert "execute_sql" in out["hint"], "should list the real tools"

    with check("execute_sql returns rows plus the SQL that produced them"):
        out = json.loads(dispatch("execute_sql", {"sql": "SELECT * FROM sales LIMIT 2"}))
        assert out["ok"] is True, out
        assert "rows" in out and "row_count" in out
        assert out["sql"].lower().startswith("select")

    with check("empty SQL is rejected with an actionable message"):
        out = json.loads(dispatch("execute_sql", {"sql": "   "}))
        assert out["ok"] is False and "No SQL" in out["error"]

    with check("missing 'sql' argument is reported, not raised"):
        out = json.loads(dispatch("execute_sql", {}))
        assert out["ok"] is False, out

    if not USE_REAL_DB:
        section("Safety wiring (stubbed database)")

        with check("a rejected statement never reaches the executor"):
            calls.clear()
            out = json.loads(dispatch("execute_sql", {"sql": "DROP TABLE sales"}))
            assert out["ok"] is False, "DROP must be refused"
            assert "rejected" in out["error"].lower()
            assert calls == ["validate"], f"executor must not run, got {calls}"

        with check("an accepted statement runs validate BEFORE execute"):
            calls.clear()
            dispatch("execute_sql", {"sql": "SELECT 1"})
            assert calls == ["validate", "execute"], calls

        with check("a database failure degrades to error text"):
            def _boom(sql: str):
                raise ConnectionError("could not connect to server")

            executor.execute_query = _boom  # type: ignore[assignment]
            out = json.loads(dispatch("execute_sql", {"sql": "SELECT 1"}))
            assert out["ok"] is False and "could not connect" in out["error"]
            executor.execute_query = _fake_execute  # type: ignore[assignment]
    else:
        section("Live database")
        print(dispatch("get_schema")[:800])
        print(dispatch("execute_sql", {"sql": "SELECT 1 AS ok"}))

    report("mcp/tools.py" + (" [--real]" if USE_REAL_DB else " [stubbed DB]"))
