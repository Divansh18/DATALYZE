"""Tiny self-check harness for the `python -m app.<module>` blocks.

Development only — nothing in the request path imports this. It exists so each
module can be run on its own and report PASS/FAIL without pulling in pytest.

Usage inside a module::

    if __name__ == "__main__":
        from app.dev import check, report

        with check("does the thing"):
            assert thing() == expected

        report("my/module.py")

Exit code is 0 when everything passes and 1 otherwise, so these can be dropped
into CI as-is.
"""

import sys
import traceback
from contextlib import contextmanager
from typing import List, Tuple

_results: List[Tuple[str, str]] = []  # (status, label)

_PASS = "ok  "
_FAIL = "FAIL"
_SKIP = "skip"


@contextmanager
def check(label: str, *, verbose_errors: bool = False):
    """Run one assertion block and record whether it passed.

    An exception marks the check failed and is swallowed, so a single broken
    assumption doesn't hide every check after it.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - reporting failures is the job
        _results.append((_FAIL, label))
        print(f"  {_FAIL}  {label}")
        print(f"        {type(exc).__name__}: {exc}")
        if verbose_errors:
            traceback.print_exc()
    else:
        _results.append((_PASS, label))
        print(f"  {_PASS}  {label}")


def skip(label: str, reason: str) -> None:
    """Record a check that could not run (missing dependency, no API key)."""
    _results.append((_SKIP, label))
    print(f"  {_SKIP}  {label}")
    print(f"        {reason}")


def section(title: str) -> None:
    print(f"\n{title}")


def report(module_name: str) -> None:
    """Print the tally and exit non-zero if anything failed."""
    passed = sum(1 for status, _ in _results if status == _PASS)
    failed = sum(1 for status, _ in _results if status == _FAIL)
    skipped = sum(1 for status, _ in _results if status == _SKIP)

    print()
    print("-" * 68)
    summary = f"{module_name}: {passed} passed"
    if failed:
        summary += f", {failed} FAILED"
    if skipped:
        summary += f", {skipped} skipped"
    print(summary)
    print("-" * 68)

    sys.exit(1 if failed else 0)
