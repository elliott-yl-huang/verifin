"""Skills: markdown playbooks with YAML-ish frontmatter, progressively disclosed to the LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

SKILL_DIR = Path(__file__).parent / "skills"


@dataclass
class Skill:
    name: str
    description: str
    triggers: List[str]
    tools: List[str]
    body: str


def _parse(text: str) -> Skill:
    _, fm, body = text.split("---", 2)
    meta: Dict[str, object] = {}
    for line in fm.strip().splitlines():
        k, v = line.split(":", 1)
        v = v.strip()
        meta[k.strip()] = [x.strip() for x in v[1:-1].split(",")] if v.startswith("[") else v
    return Skill(meta["name"], meta["description"], meta.get("triggers", []), meta.get("tools", []), body.strip())


def load_skills(path: Path = SKILL_DIR) -> Dict[str, Skill]:
    return {s.name: s for s in (_parse(p.read_text()) for p in sorted(path.glob("*.md")))}


def route(question: str, skills: Dict[str, Skill]) -> str:
    """Keyword router used offline (and as a hint to the LLM). Longer trigger phrases weigh more."""
    q = question.lower()
    scores = {n: sum(len(t.split()) for t in s.triggers if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", q))
              for n, s in skills.items()}
    best = max(scores, key=lambda n: scores[n])
    return best if scores[best] else "adhoc-sql"


def skill_index(skills: Dict[str, Skill]) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
