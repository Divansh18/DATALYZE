"""Shared helper logic used by MCP tools.

Single responsibility: turn raw Python values into something safe to hand back
to Claude, and turn exceptions into error text Claude can act on.

This module does not register tools, does not start the MCP server, and does
not touch the database.
"""

import json
import logging
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Sequence
from uuid import UUID

logger = logging.getLogger(__name__)

# A result set has to fit in Claude's context and stay cheap. Rows beyond this
# are dropped and the truncation is reported so Claude knows to add a LIMIT or
# aggregate instead of assuming it saw everything.
MAX_ROWS = 200

# Guard against a single wide text column blowing up the payload.
MAX_CELL_CHARS = 2000


class ToolExecutionError(Exception):
    """Raised by a tool when it cannot produce a result.

    The message is shown to Claude verbatim, so make it actionable: say what
    was wrong with the input, not just that something failed.
    """


def to_jsonable(value: Any) -> Any:
    """Convert a psycopg2/SQLAlchemy value into something `json.dumps` accepts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        # float() is lossy for money; str() keeps precision and Claude reads
        # either fine. Precision matters more here than JSON number typing.
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return str(value)


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + f"... [truncated, {len(value)} chars]"
    return value


def format_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Shape query results into the payload handed back to Claude.

    Returns row count, whether the output was truncated, the column names, and
    the rows themselves. `row_count` is the true count before truncation so
    Claude never reports a wrong total.
    """
    total = len(rows)
    visible = list(rows[:MAX_ROWS])

    payload: Dict[str, Any] = {
        "row_count": total,
        "truncated": total > MAX_ROWS,
        "columns": list(visible[0].keys()) if visible else [],
        "rows": [
            {str(k): _clip(to_jsonable(v)) for k, v in row.items()} for row in visible
        ],
    }

    if payload["truncated"]:
        payload["note"] = (
            f"Only the first {MAX_ROWS} of {total} rows are shown. Re-run with an "
            f"explicit LIMIT or an aggregate query if you need the full picture."
        )

    return payload


def success(data: Any) -> str:
    """Serialize a successful tool result."""
    return json.dumps({"ok": True, **_as_mapping(data)}, ensure_ascii=False, indent=2)


def failure(message: str, *, hint: str = "") -> str:
    """Serialize a failed tool result.

    Errors are returned rather than raised so Claude can read what went wrong
    and retry with a corrected query instead of the whole turn dying.
    """
    payload: Dict[str, Any] = {"ok": False, "error": message}
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _as_mapping(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    return {"result": to_jsonable(data)}


def run_tool(name: str, fn, *args, **kwargs) -> str:
    """Execute a tool function and normalize its outcome.

    Every tool goes through here so error handling lives in one place:
    expected failures (bad SQL, rejected statement) come back as readable text,
    unexpected ones are logged with a traceback and reported without leaking
    internals to the model.
    """
    try:
        return success(fn(*args, **kwargs))
    except ToolExecutionError as exc:
        logger.info("Tool %s rejected the request: %s", name, exc)
        return failure(str(exc))
    except Exception as exc:  # noqa: BLE001 - a tool must never kill the loop
        logger.exception("Tool %s failed unexpectedly", name)
        return failure(
            f"{type(exc).__name__}: {exc}",
            hint="This looks like a database or connection problem, not a bad query.",
        )


def describe_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Pass schema information through the same JSON-safety pass as rows."""
    return {"schema": to_jsonable(schema)}


__all__ = [
    "MAX_ROWS",
    "ToolExecutionError",
    "describe_schema",
    "failure",
    "format_rows",
    "run_tool",
    "success",
    "to_jsonable",
]


# ---------------------------------------------------------------------------
# Self-check:  python -m app.features.mcp.service      (run from backend/)
#
# Pure functions, no database and no network — this always runs.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from app.dev import check, report

    logging.basicConfig(level=logging.CRITICAL)  # silence the expected traceback

    with check("Decimal keeps full precision (money must not become a float)"):
        assert to_jsonable(Decimal("1234.56")) == "1234.56"

    with check("dates/UUID/bytes/None are JSON-safe"):
        assert to_jsonable(date(2026, 7, 1)) == "2026-07-01"
        assert to_jsonable(datetime(2026, 7, 1, 9, 30)) == "2026-07-01T09:30:00"
        assert to_jsonable(UUID("00000000-0000-0000-0000-000000000001")).endswith("001")
        assert to_jsonable(b"abc") == "<3 bytes>"
        assert to_jsonable(None) is None

    with check("nested dict/list values are converted recursively"):
        out = to_jsonable({"totals": [Decimal("1"), {"d": date(2026, 1, 1)}]})
        assert out == {"totals": ["1", {"d": "2026-01-01"}]}, out

    with check("format_rows reports columns and an exact row count"):
        out = format_rows([{"product": "Widget", "revenue": Decimal("10.50")}])
        assert out["columns"] == ["product", "revenue"], out
        assert out["row_count"] == 1 and out["truncated"] is False
        assert out["rows"][0]["revenue"] == "10.50"

    with check(f"over {MAX_ROWS} rows -> truncated, but row_count stays true"):
        out = format_rows([{"n": i} for i in range(MAX_ROWS + 25)])
        assert out["row_count"] == MAX_ROWS + 25, "true total must survive"
        assert len(out["rows"]) == MAX_ROWS, "payload must be capped"
        assert out["truncated"] is True and "note" in out

    with check("empty result set does not crash"):
        out = format_rows([])
        assert out == {"row_count": 0, "truncated": False, "columns": [], "rows": []}

    with check(f"a giant cell is clipped at {MAX_CELL_CHARS} chars"):
        out = format_rows([{"blob": "x" * 5000}])
        assert "truncated" in out["rows"][0]["blob"]
        assert len(out["rows"][0]["blob"]) < 5000

    with check("success()/failure() emit valid JSON with an ok flag"):
        assert json.loads(success({"row_count": 0}))["ok"] is True
        bad = json.loads(failure("nope", hint="try again"))
        assert bad["ok"] is False and bad["error"] == "nope" and bad["hint"] == "try again"

    with check("run_tool turns a ToolExecutionError into readable text"):
        def _reject():
            raise ToolExecutionError("Only SELECT is allowed.")

        out = json.loads(run_tool("t", _reject))
        assert out["ok"] is False and "Only SELECT" in out["error"]

    with check("run_tool never lets an unexpected exception escape"):
        def _explode():
            raise RuntimeError("connection reset")

        out = json.loads(run_tool("t", _explode))
        assert out["ok"] is False and "connection reset" in out["error"]

    report("mcp/service.py")
