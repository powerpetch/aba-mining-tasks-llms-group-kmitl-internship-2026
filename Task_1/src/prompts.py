from __future__ import annotations

from pathlib import Path


def load_prompt(repo_root: Path, relative_path: str) -> str:
    path = repo_root / relative_path
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def build_modular_prompt(repo_root: Path, rules: list[int]) -> str:
    """Assemble a prompt from header + selected rule files + footer.

    The assembled string still contains {{TOPICS}}, {{TOPIC_COUNT}}, and
    {{REVIEW_TEXT}} placeholders — call render_prompt() on the result before
    sending to the model.

    Args:
        repo_root: Absolute path to the ABA_Mining directory.
        rules: Ordered list of rule numbers to include, e.g. [1, 2, 5].
    """
    rules_dir = repo_root / "prompts" / "task1" / "rules"
    header = (repo_root / "prompts" / "task1" / "header.txt").read_text(encoding="utf-8")
    footer = (repo_root / "prompts" / "task1" / "footer.txt").read_text(encoding="utf-8")

    rule_blocks: list[str] = []
    for r in rules:
        rule_file = rules_dir / f"rule{r}.txt"
        if not rule_file.exists():
            raise FileNotFoundError(f"Rule file not found: {rule_file}")
        rule_blocks.append(rule_file.read_text(encoding="utf-8").strip())

    return header.rstrip() + "\n\n" + "\n\n".join(rule_blocks) + "\n" + footer
