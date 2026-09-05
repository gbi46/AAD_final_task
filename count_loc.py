"""LOC-лічильник для S15 (Fair Comparison, `claude_plan/05_BONUS.md`).

Одне фіксоване правило, застосоване ОДНАКОВО до обох фреймворків: LOC =
кількість рядків файлу, що НЕ є ані порожніми, ані рядком, який (після
`.strip()`) складається ЛИШЕ з коментаря (`#...`). Docstring-рядки
рахуються (так само, як і в обох файлах, і як типово роблять прості
LOC-лічильники) — не виключаються окремо, бо відділити "докстрінг як
документація" від "докстрінг як частина коду" було б довільним рішенням
для .py-файлів такого стилю (тут докстрінги документують нетривіальні
архітектурні рішення, це частина вартості реалізації, не шум).

Категорії (план: "framework-specific файли окремо від shared code, без
tests/docs/generated/venv"):
- LangGraph-specific: mas_langgraph.py, hitl.py (HITL, S09, — можливість,
  якої немає в mas_crewai.py, S14 docstring пояснює чому).
- CrewAI-specific: mas_crewai.py.
- Shared (НЕ зараховується до жодного фреймворку -- обидва перевикористовують
  БЕЗ ЗМІН): tools_legacy.py, knowledge.py, guardrails.py, safety.py,
  trajectory_logger.py, mcp_server.py.
- Явно ВИКЛЮЧЕНО з порівняння: tests/*, *.md, observability.py/evals.py/
  red_team.py/count_loc.py (S10-S13/S15 — крос-функціональний
  інструментарій, не частина самої workflow-реалізації жодного
  фреймворку), усе згенероване (runtime/, chroma_db/, *.json результати,
  trajectory.json), venv (.venv/, .venv-crewai/), requirements*.txt,
  .gitignore, .env*, docs/.

Важлива асиметрія обсягу (чесно, не приховано): `mas_langgraph.py`
реалізує ПОВНУ S05-S09 функціональність (4 агенти, guardrail-інтеграція,
HITL-виклики, persistence-хуки) — усе, що вимагав план для просунутого
рівня. `mas_crewai.py` (S14, Bonus) реалізує ЛИШЕ 3 спеціалісти +
coordinator без HITL/persistence (план S14 не вимагає їх відтворювати).
Тому пряме LOC-порівняння НЕ означає "CrewAI ефективніший" -- це різний
обсяг вимог, а не різна густина коду на еквівалентний функціонал. Розбір
— у README (S15).
"""

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
        "note": (
            "mas_langgraph.py includes full S05-S09 scope (4 agents, guardrail "
            "integration, HITL, persistence hooks); mas_crewai.py (S14, Bonus) "
            "implements only 3 specialists + coordinator, no HITL/persistence "
            "(not required by S14). Raw LOC difference reflects scope "
            "difference, not code density on equivalent functionality -- see "
            "README S15 analysis."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
