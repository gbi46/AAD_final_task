"""Підрахунок непорожніх рядків без окремих рядків-коментарів.

Docstrings враховуються. Код фреймворків і спільні модулі рахуються окремо;
тести, документація та згенеровані дані виключені. Реалізації відрізняються
обсягом: HITL і persistence є лише у LangGraph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

LANGGRAPH_SPECIFIC = ["mas_langgraph.py", "hitl.py"]
CREWAI_SPECIFIC = ["mas_crewai.py"]
SHARED = ["tools_legacy.py", "knowledge.py", "guardrails.py", "safety.py", "trajectory_logger.py", "mcp_server.py"]


def count_loc(path: Path) -> int:
    """Рядки файлу мінус порожні мінус чисто коментарні (після strip)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def _sum_group(names: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in names:
        path = _ROOT / name
        if not path.exists():
            result[name] = 0
            continue
        result[name] = count_loc(path)
    return result


def main() -> None:
    langgraph = _sum_group(LANGGRAPH_SPECIFIC)
    crewai = _sum_group(CREWAI_SPECIFIC)
    shared = _sum_group(SHARED)

    report = {
        "rule": "non-blank, non-comment-only lines (line.strip() truthy and not startswith '#')",
        "excluded": ["tests/*", "*.md", "observability.py", "evals.py", "red_team.py", "count_loc.py", "generated artifacts", "venvs", "requirements*.txt", ".env*", "docs/"],
        "langgraph_specific": {"files": langgraph, "total": sum(langgraph.values())},
        "crewai_specific": {"files": crewai, "total": sum(crewai.values())},
        "shared_reused_by_both": {"files": shared, "total": sum(shared.values())},
        "note": "LangGraph includes HITL and persistence; CrewAI does not. LOC reflects different scope.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
