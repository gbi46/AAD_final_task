"""Порівняння LangGraph і CrewAI на чотирьох однакових live-запитах через MCP.

Модель і температура однакові; один прогрів виключено з результатів.
На сценарій виконується один вимір. LangGraph отримує токени з LangSmith,
CrewAI — з usage metrics. Час розробки не вимірювався."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_RUNTIME_DIR = _ROOT / "runtime" / "comparison"
_RESULTS_PATH = _ROOT / "comparison_results.json"
_COMPARISON_PROJECT = "aad-final-mas-comparison"

WARMUP_QUERY = "Привіт!"

# Той самий текст, що й evals.py -- жодного окремого набору fixtures.
QUERIES = [
    ("EVAL-01", "simple", "Оціни бюджет Львова на 3 дні для двох"),
    ("EVAL-02", "multi-step", "У Львові дощ: перевір погоду і підбери культурні активності"),
    ("EVAL-03", "RAG-heavy", "Які правила відвідування Лувру?"),
    ("EVAL-04", "cross-agent", "У Львові дощ: зміни маршрут і перерахуй бюджет на 3 дні для двох"),
]


# --- LangGraph -----------------------------------------------------------


async def _run_langgraph() -> list[dict[str, Any]]:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
    import observability

    enabled = observability.configure_observability(project=_COMPARISON_PROJECT)
    if not enabled:
        raise RuntimeError("LANGSMITH_API_KEY відсутній -- Неможливо виміряти tokens для LangGraph без нього.")

    from tools_legacy import configure_storage

    import mas_langgraph as m

    runtime_dir = _RUNTIME_DIR / "langgraph"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    configure_storage(runtime_dir / "chroma_db", runtime_dir / "bookings.json")

    mas = m.MASApp(provider="openrouter", run_id="framework-comparison")
    await mas.connect_mcp(
        env={
            **os.environ,
            "MCP_KNOWLEDGE_PATH": str(runtime_dir / "chroma_db"),
            "MCP_BOOKING_PATH": str(runtime_dir / "bookings.json"),
        }
    )

    # Warm-up (відкидається).
    await mas.handle_query(WARMUP_QUERY, thread_id="comparison-warmup")

    records = []
    for scenario_id, scenario_type, query in QUERIES:
        thread_id = f"comparison-{scenario_id.lower()}"
        started = time.monotonic()
        state = await mas.handle_query(query, thread_id=thread_id)
        latency_ms = int((time.monotonic() - started) * 1000)
        trajectory = state.get("trajectory", [])
        agents_used = []
        for entry in trajectory:
            name = entry.get("agent_name")
            if name and name != "supervisor" and name not in agents_used:
                agents_used.append(name)
        tools_called = []
        for entry in trajectory:
            if entry.get("node_name") != "tool_dispatch":
                continue
            for call in entry.get("tool_calls") or []:
                if call.get("name"):
                    tools_called.append(call["name"])
        results_text = " | ".join(state.get("results", []))
        records.append(
            {
                "scenario_id": scenario_id,
                "type": scenario_type,
                "query": query,
                "framework": "langgraph",
                "thread_id": thread_id,
                "latency_ms": latency_ms,
                "agents_used": agents_used,
                "tools_called": tools_called,
                "answer": results_text,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
        )

    # Токени -- через LangSmith SDK ПІСЛЯ всіх запитів. Зіставлення СТРОГО
    # за `thread_id` у `run.extra["metadata"]["thread_id"]` (LangGraph сам
    # проставляє його з `MASApp.config(thread_id)` у кожен root run) -- НЕ
    # за start_time/приблизним вікном: перша версія цього скрипта саме так
    # і робила (нечітке зіставлення за часом), і це дало ЗСУВ НА ОДНУ
    # позицію між сценаріями (перевірено emпірично -- порівняно з прямим
    # запитом `run.extra["metadata"]["thread_id"]` у LangSmith UI/SDK,
    # кожен запис виявився зміщеним відносно свого справжнього trace).
    # thread_id -- єдине надійне джерело істини для цього зіставлення.
    await asyncio.sleep(2.0)  # дати LangSmith час доскласти останні traces
    from langsmith import Client

    client = Client()
    roots = list(client.list_runs(project_name=_COMPARISON_PROJECT, is_root=True, limit=50))
    roots_by_thread = {(r.extra or {}).get("metadata", {}).get("thread_id"): r for r in roots}
    for record in records:
        match = roots_by_thread.get(record["thread_id"])
        if match is None:
            continue
        tree = list(client.list_runs(project_name=_COMPARISON_PROJECT, trace_id=match.trace_id))
        prompt = sum(r.prompt_tokens or 0 for r in tree if r.run_type == "llm")
        completion = sum(r.completion_tokens or 0 for r in tree if r.run_type == "llm")
        total = sum(r.total_tokens or 0 for r in tree if r.run_type == "llm")
        record["prompt_tokens"] = prompt
        record["completion_tokens"] = completion
        record["total_tokens"] = total
        record["trace_id"] = str(match.trace_id)

    return records


# --- CrewAI ----------------------------------------------------------------


def _run_crewai() -> list[dict[str, Any]]:
    crewai_python = _ROOT / ".venv-crewai" / "bin" / "python"
    if not crewai_python.exists():
        raise RuntimeError(f"{crewai_python} не знайдено -- створіть .venv-crewai (див. README).")

    runtime_dir = _RUNTIME_DIR / "crewai"

    def _bench(query: str, tid: str) -> dict[str, Any]:
        out = subprocess.run(
            [str(crewai_python), str(_ROOT / "mas_crewai_bench.py"), query, str(runtime_dir / tid)],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            timeout=300,
        )
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"mas_crewai_bench.py не повернув валідний JSON. stderr:\n{out.stderr[-4000:]}") from exc

    # Warm-up (відкидається).
    _bench(WARMUP_QUERY, "warmup")

    records = []
    for scenario_id, scenario_type, query in QUERIES:
        result = _bench(query, scenario_id.lower())
        records.append(
            {
                "scenario_id": scenario_id,
                "type": scenario_type,
                "query": query,
                "framework": "crewai",
                "latency_ms": result.get("latency_ms"),
                "agents_used": result.get("agents_used", []),
                "tools_called": result.get("tools_called", []),
                "answer": result.get("answer", ""),
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "total_tokens": result.get("total_tokens"),
                "successful_requests": result.get("successful_requests"),
            }
        )
    return records


# --- Оркестрація -------------------------------------------------------------


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"

    langgraph_records: list[dict[str, Any]] = []
    crewai_records: list[dict[str, Any]] = []

    if which in ("both", "langgraph"):
        langgraph_records = asyncio.run(_run_langgraph())
    if which in ("both", "crewai"):
        crewai_records = _run_crewai()

    payload = {
        "methodology": {
            "queries_source": "evals.py EVAL-01..04 (той самий текст, ті самі fixtures)",
            "model": os.getenv("OPENROUTER_MODEL", "google/gemini-3.6-flash"),
            "temperature": 0.1,
            "mcp_mode": "real MCP stdio subprocess for BOTH frameworks (MASApp.connect_mcp() for LangGraph, crewai_tools.MCPServerAdapter for CrewAI)",
            "warm_up": f"1 discarded call ({WARMUP_QUERY!r}) per framework before measured queries",
            "repeats": "N=1 per scenario per framework (live LLM calls cost real tokens; repeats were not run -- documented limitation, not hidden)",
            "dev_time": None,
            "dev_time_note": "Development time was not recorded for either framework.",
            "loc": "see count_loc.py output / README comparison table",
        },
        "langgraph": langgraph_records,
        "crewai": crewai_records,
    }
    _RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записано {_RESULTS_PATH}")
    for rec in langgraph_records + crewai_records:
        print(
            f"  [{rec['framework']}] {rec['scenario_id']}: latency={rec['latency_ms']}ms "
            f"tokens={rec.get('total_tokens')} agents={rec.get('agents_used')}"
        )


if __name__ == "__main__":
    main()
