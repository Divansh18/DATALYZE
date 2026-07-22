import os
from pathlib import Path
from typing import Dict, Iterable, Optional


# backend/app/llm/prompt_loader.py -> parents[2] == backend/
DEFAULT_PROMPT_FOLDER = Path(__file__).resolve().parents[2] / "prompts"


def resolve_prompt_folder(prompt_folder: Optional[Path] = None) -> Path:
    if prompt_folder is not None:
        return prompt_folder

    env_folder = os.getenv("PROMPT_FOLDER")
    if env_folder:
        return Path(env_folder).expanduser().resolve()

    return DEFAULT_PROMPT_FOLDER


def load_prompt(prompt_name: str, prompt_folder: Optional[Path] = None) -> str:
    prompt_dir = resolve_prompt_folder(prompt_folder)
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt folder does not exist: {prompt_dir}")

    prompt_file = prompt_dir / prompt_name
    if not prompt_file.suffix:
        prompt_file = prompt_file.with_suffix(".txt")

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")

    return prompt_file.read_text(encoding="utf-8").strip()


def load_prompts(prompt_names: Iterable[str], prompt_folder: Optional[Path] = None) -> Dict[str, str]:
    return {name: load_prompt(name, prompt_folder=prompt_folder) for name in prompt_names}


def list_prompts(prompt_folder: Optional[Path] = None) -> Dict[str, str]:
    prompt_dir = resolve_prompt_folder(prompt_folder)
    if not prompt_dir.exists():
        return {}

    return {
        prompt_file.stem: prompt_file.read_text(encoding="utf-8").strip()
        for prompt_file in prompt_dir.glob("*.txt")
    }


# ---------------------------------------------------------------------------
# Self-check:  python -m app.llm.prompt_loader        (run from backend/)
#
# Reads the real prompts/ folder, so it also tells you which prompt files are
# still empty.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    from app.dev import check, report, section

    EXPECTED = ["system_prompt", "sql_generation", "analysis", "report"]

    section(f"Default folder: {DEFAULT_PROMPT_FOLDER}")

    with check("default folder resolves to backend/prompts and exists"):
        assert DEFAULT_PROMPT_FOLDER.name == "prompts"
        assert DEFAULT_PROMPT_FOLDER.parent.name == "backend"
        assert DEFAULT_PROMPT_FOLDER.exists(), f"missing: {DEFAULT_PROMPT_FOLDER}"

    with check(f"all four prompt files are present: {', '.join(EXPECTED)}"):
        found = set(list_prompts())
        missing = [name for name in EXPECTED if name not in found]
        assert not missing, f"missing files: {missing}"

    with check("no prompt file is empty (an empty prompt silently breaks Claude)"):
        empty = [name for name, text in list_prompts().items() if not text.strip()]
        assert not empty, f"empty prompt files: {empty}"

    section("Loading")

    with check("load_prompt appends .txt when no suffix is given"):
        assert load_prompt("system_prompt") == load_prompt("system_prompt.txt")

    with check("load_prompt strips surrounding whitespace"):
        text = load_prompt("system_prompt")
        assert text == text.strip()

    with check("load_prompts returns every requested prompt"):
        loaded = load_prompts(EXPECTED)
        assert sorted(loaded) == sorted(EXPECTED), sorted(loaded)

    with check("a missing prompt raises FileNotFoundError naming the file"):
        try:
            load_prompt("does_not_exist")
            raise AssertionError("should have raised")
        except FileNotFoundError as exc:
            assert "does_not_exist" in str(exc)

    section("Overrides")

    with check("an explicit folder argument wins"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "custom.txt").write_text("  override  ", encoding="utf-8")
            assert load_prompt("custom", prompt_folder=tmp_path) == "override"

    with check("PROMPT_FOLDER env var is honoured when no argument is passed"):
        with tempfile.TemporaryDirectory() as tmp:
            saved = os.environ.get("PROMPT_FOLDER")
            os.environ["PROMPT_FOLDER"] = tmp
            try:
                assert resolve_prompt_folder() == Path(tmp).expanduser().resolve()
            finally:
                if saved is None:
                    os.environ.pop("PROMPT_FOLDER", None)
                else:
                    os.environ["PROMPT_FOLDER"] = saved

    with check("a nonexistent folder raises FileNotFoundError, not a silent miss"):
        try:
            load_prompt("system_prompt", prompt_folder=Path("no/such/folder"))
            raise AssertionError("should have raised")
        except FileNotFoundError:
            pass

    section("Contents")
    for name, text in sorted(list_prompts().items()):
        preview = text.replace("\n", " ")[:60]
        print(f"  {name:<16} {len(text):>6} chars  {preview}")

    report("llm/prompt_loader.py")
