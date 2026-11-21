"""Anthropic-style skills: folders with SKILL.md (name/description frontmatter)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SKILL_DIRS = [Path.cwd() / "skills", Path.home() / ".config" / "llmui" / "skills"]
_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)

LOAD_SKILL = "load_skill"
READ_SKILL_FILE = "read_skill_file"


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # skill directory
    body: str


def parse_skill_md(text: str) -> tuple[dict, str]:
    m = _FM.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), m.group(2)


def discover(dirs: list[Path] | None = None) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for d in dirs if dirs is not None else SKILL_DIRS:
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*/SKILL.md")):
            meta, body = parse_skill_md(md.read_text(errors="replace"))
            name = str(meta.get("name") or md.parent.name)
            if name in skills:
                continue
            skills[name] = Skill(name, str(meta.get("description", "")).strip(), md.parent, body.strip())
    return skills


def safe_path(skill: Skill, rel: str) -> Path | None:
    """Resolve rel inside the skill dir; None if it escapes."""
    root = skill.path.resolve()
    p = (root / rel).resolve()
    return p if p == root or root in p.parents else None


def prompt_section(skills: dict[str, Skill]) -> str:
    if not skills:
        return ""
    items = "\n".join(f'<skill name="{s.name}">{s.description}</skill>' for s in skills.values())
    return (
        "\n\nYou have access to skills: specialised instructions for particular tasks.\n"
        f"When a request matches a skill's description, first call `{LOAD_SKILL}` with its name "
        "and follow the instructions it returns.\n"
        f"<available_skills>\n{items}\n</available_skills>"
    )


def tool_schemas() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": LOAD_SKILL,
            "description": "Load the full instructions of a skill by name.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
        {"type": "function", "function": {
            "name": READ_SKILL_FILE,
            "description": "Read a file bundled inside a skill folder.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"}, "path": {"type": "string", "description": "Relative path in the skill folder"}},
                "required": ["name", "path"]}}},
    ]


def is_skill_tool(name: str) -> bool:
    return name in (LOAD_SKILL, READ_SKILL_FILE)


def call_tool(skills: dict[str, Skill], name: str, args: dict) -> str:
    skill = skills.get(str(args.get("name", "")))
    if skill is None:
        return f"Error: unknown skill '{args.get('name')}'. Available: {', '.join(skills) or 'none'}"
    if name == LOAD_SKILL:
        others = [str(p.relative_to(skill.path)) for p in sorted(skill.path.rglob("*"))
                  if p.is_file() and p.name != "SKILL.md"]
        extra = f"\n\nBundled files (use {READ_SKILL_FILE}):\n" + "\n".join(others) if others else ""
        return skill.body + extra
    p = safe_path(skill, str(args.get("path", "")))
    if p is None or not p.is_file():
        return "Error: file not found or outside the skill folder."
    return p.read_text(errors="replace")[:50_000]
