"""Run every module self-check in one go.

    python selftest.py            # offline only
    python selftest.py --live     # also make real Anthropic API calls

Each module is run in its own subprocess so one crash can't hide the rest.
Exits non-zero if any module fails, so this works as a CI step.
"""

import subprocess
import sys

MODULES = [
    ("app.llm.prompt_loader", []),
    ("app.llm.claude", ["--live"] if "--live" in sys.argv else []),
    ("app.features.mcp.service", []),
    ("app.features.mcp.tools", []),
    ("app.features.mcp.server", ["--selftest"]),
]


def main() -> int:
    failed = []

    for module, args in MODULES:
        # flush=True matters: without it the parent's buffered output lands
        # after the subprocess's unbuffered output and the headers scramble.
        print("=" * 68, flush=True)
        print(f"  python -m {module} {' '.join(args)}".rstrip(), flush=True)
        print("=" * 68, flush=True)
        result = subprocess.run([sys.executable, "-m", module, *args])
        if result.returncode != 0:
            failed.append(module)
        print(flush=True)

    print("=" * 68)
    if failed:
        print(f"FAILED ({len(failed)}/{len(MODULES)}): {', '.join(failed)}")
    else:
        print(f"All {len(MODULES)} module self-checks passed.")
    print("=" * 68)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
