"""Оцінювання MAS: simple, multi-step, RAG, handoff і HITL-сценарії.

Режим demo не потребує LLM; live використовує OpenRouter.
Результати, час, агенти й інструменти записуються в eval_results.json.
trace_id може бути null: локальний collect_runs не гарантує публікацію trace."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from tools_legacy import configure_storage

_ROOT = Path(__file__).resolve().parent
_RUNTIME_DIR = _ROOT / "runtime" / "evals"
_DB_PATH = _RUNTIME_DIR / "evals_hitl.db"
_KNOWLEDGE_PATH = _RUNTIME_DIR / "chroma_db"
_BOOKING_PATH = _RUNTIME_DIR / "bookings.json"
_RESULTS_PATH = _ROOT / "eval_results.json"


# --- Результат одного сценарію -----------------------------------------------


@dataclass
class EvalRecord:
    scenario_id: str
    scenario_type: str
    query: str
    status: str  # "pass" | "fail" | "blocked"
    latency_ms: int
    agents_used: list[str]
    tools_called: list[str]
    mode: str
    expected: str
    observed: str
    error: str | None = None
    trace_id: str | None = None
    human_wait_ms: int | None = None
    model: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "type": self.scenario_type,
            "query": self.query,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "agents_used": self.agents_used,
            "tools_called": self.tools_called,
            "mode": self.mode,
            "expected": self.expected,
            "observed": self.observed,
            "error": self.error,
            "trace_id": self.trace_id,
            "human_wait_ms": self.human_wait_ms,
            "model": self.model,
            "versions": self.versions,
            "timestamp": self.timestamp,
        }


# --- Допоміжне: витяг agents_used/tools_called із реальної траєкторії -------


def _agents_used(trajectory: list[dict[str, Any]]) -> list[str]:
    """Впорядкований (без дублів) список СПЕЦІАЛІЗОВАНИХ агентів, що реально
    обробляли запит — `supervisor`-бухгалтерія (маршрутизація/handoff)
    навмисно виключена, бо це не "агент, який опрацював запит", а router.
    """
    seen: list[str] = []
    for entry in trajectory:
        name = entry.get("agent_name")
        if name and name != "supervisor" and name not in seen:
            seen.append(name)
    return seen


def _tools_called(trajectory: list[dict[str, Any]]) -> list[str]:
    """Список РЕАЛЬНО виконаних tool-викликів (вузол `tool_dispatch` у
    `MASApp.dispatch_tool` — ЄДИНА точка виконання, і local, і MCP tools),
    у порядку виконання. Навмисно НЕ використовує `*_agent`/`*_executor`
    записи (там лише НАМІР викликати tool, не підтверджене виконання) —
    інакше ризиковий `book_hotel`, що чекає approval_gate, зарахувався б як
    "викликаний" ще до реального dispatch.
    """
    called: list[str] = []
    for entry in trajectory:
        if entry.get("node_name") != "tool_dispatch":
            continue
        for call in entry.get("tool_calls") or []:
            name = call.get("name")
            if name:
                called.append(name)
    return called


CheckFn = Callable[[dict[str, Any]], tuple[bool, str]]


async def _run_query_scenario(
    scenario_id: str, query: str, provider: str, check: CheckFn
) -> dict[str, Any]:
    """Спільний прогін для EVAL-01..04: свіжий MASApp (нова сесія), один
    `handle_query`, перевірка через `check(final_state)`.
    """
    import mas_langgraph as m

    mas = m.MASApp(provider=provider, run_id=f"eval-{scenario_id.lower()}")
    thread_id = f"eval-{scenario_id.lower()}"
    started = time.monotonic()
    final_state = await mas.handle_query(query, thread_id=thread_id)
    latency_ms = int((time.monotonic() - started) * 1000)
    trajectory = final_state.get("trajectory", [])
    try:
        ok, observed = check(final_state)
        error = None if ok else observed
    except Exception as exc:  # захист від несподіваного KeyError/формату стану
        ok, observed, error = False, f"Виключення під час перевірки: {exc}", str(exc)
    return {
        "status": "pass" if ok else "fail",
        "observed": observed,
        "error": error,
        "agents_used": _agents_used(trajectory),
        "tools_called": _tools_called(trajectory),
        "latency_ms": latency_ms,
        "human_wait_ms": None,
    }


# --- EVAL-01: simple ---------------------------------------------------------

EVAL01_QUERY = "Оціни бюджет Львова на 3 дні для двох"
EVAL01_EXPECTED = "billing, calculate_trip_cost, правильна сума (52 EUR/день * 3 дн. * 2 ос. = 312 EUR)"


def _check_eval01(state: dict[str, Any]) -> tuple[bool, str]:
    results = " | ".join(state.get("results", []))
    tools = _tools_called(state.get("trajectory", []))
    agent = state.get("current_agent")
    ok = agent == "billing" and "calculate_trip_cost" in tools and "312" in results and "Львів" in results
    return ok, f"current_agent={agent}, tools={tools}, results={results!r}"


# --- EVAL-02: multi-step ------------------------------------------------------

EVAL02_QUERY = "У Львові дощ: перевір погоду і підбери культурні активності"
EVAL02_EXPECTED = "tech, get_weather + find_activities (≥2 tool steps)"


def _check_eval02(state: dict[str, Any]) -> tuple[bool, str]:
    results = " | ".join(state.get("results", []))
    tools = _tools_called(state.get("trajectory", []))
    agent = state.get("current_agent")
    ok = (
        agent == "tech"
        and "get_weather" in tools
        and "find_activities" in tools
        and len(tools) >= 2
        and "дощ" in results
    )
    return ok, f"current_agent={agent}, tools={tools}, results={results!r}"


# --- EVAL-03: RAG-heavy -------------------------------------------------------

EVAL03_QUERY = "Які правила відвідування Лувру?"
EVAL03_EXPECTED = "researcher, search_knowledge, посилання на документ (напр. [travel-04])"


def _check_eval03(state: dict[str, Any]) -> tuple[bool, str]:
    results = " | ".join(state.get("results", []))
    tools = _tools_called(state.get("trajectory", []))
    agent = state.get("current_agent")
    import re as _re

    has_citation = bool(_re.search(r"\[travel-\d\d\]", results))
    ok = agent == "researcher" and "search_knowledge" in tools and has_citation and "Лувр" in results
    return ok, f"current_agent={agent}, tools={tools}, has_citation={has_citation}, results={results!r}"


# --- EVAL-04: cross-agent ------------------------------------------------------

EVAL04_QUERY = "У Львові дощ: зміни маршрут і перерахуй бюджет на 3 дні для двох"
EVAL04_EXPECTED = "tech -> billing (реальний handoff), збережений контекст (3 дн./2 ос.), фінальний бюджет"


def _check_eval04(state: dict[str, Any]) -> tuple[bool, str]:
    results = " | ".join(state.get("results", []))
    tools = _tools_called(state.get("trajectory", []))
    agents = _agents_used(state.get("trajectory", []))
    handoffs = state.get("handoff_count", 0)
    ok = (
        agents == ["tech", "billing"]
        and handoffs == 1
        and "get_weather" in tools
        and "calculate_trip_cost" in tools
        and "312" in results
    )
    return ok, f"agents_used={agents}, handoff_count={handoffs}, tools={tools}, results={results!r}"


# --- EVAL-05: HITL -------------------------------------------------------------

EVAL05_QUERY = "Забронюй Demo Hotel у Львові на 2 ночі за 120 EUR з 2026-10-01"
EVAL05_EXPECTED = "billing -> interrupt (жодного booking) -> approve -> рівно один booking"


async def _run_eval05(provider: str) -> dict[str, Any]:
    """Перевірити HITL з файловим checkpoint і скриптованим approve.

    latency_ms вимірює виконання графа, human_wait_ms — проміжок між
    interrupt і resume; рішення в цьому сценарії надходить автоматично."""
    import mas_langgraph as m

    thread_id = "eval-05"
    async with AsyncSqliteSaver.from_conn_string(str(_DB_PATH)) as saver:
        mas = m.MASApp(provider=provider, run_id="eval-05", checkpointer=saver)

        before_count = len(json.loads(_BOOKING_PATH.read_text())) if _BOOKING_PATH.exists() else 0

        t0 = time.monotonic()
        paused_state = await mas.handle_query(EVAL05_QUERY, thread_id=thread_id)
        t1 = time.monotonic()
        interrupts = paused_state.get("__interrupt__")
        if not interrupts:
            return {
                "status": "fail",
                "observed": "Граф НЕ зупинився на interrupt() — очікувався pending_approval для book_hotel.",
                "error": "no interrupt",
                "agents_used": _agents_used(paused_state.get("trajectory", [])),
                "tools_called": _tools_called(paused_state.get("trajectory", [])),
                "latency_ms": int((t1 - t0) * 1000),
                "human_wait_ms": None,
            }
        payload = interrupts[0].value
        after_interrupt_count = len(json.loads(_BOOKING_PATH.read_text())) if _BOOKING_PATH.exists() else 0

        t2 = time.monotonic()
        final_state = await mas.app.ainvoke(Command(resume={"action": "approve"}), config=mas.config(thread_id))
        t3 = time.monotonic()

        after_approve_count = len(json.loads(_BOOKING_PATH.read_text())) if _BOOKING_PATH.exists() else 0
        results = " | ".join(final_state.get("results", []))
        trajectory = final_state.get("trajectory", [])
        tools = _tools_called(trajectory)

        ok = (
            payload.get("tool") == "book_hotel"
            and payload.get("agent_name") == "billing"
            and bool(payload.get("operation_id"))
            and after_interrupt_count == before_count  # ДО рішення людини — жодного booking
            and after_approve_count == before_count + 1  # ПІСЛЯ approve — рівно один
            and "book_hotel" in tools
            and "BOOK-" in results
        )
        observed = (
            f"before={before_count}, after_interrupt={after_interrupt_count}, "
            f"after_approve={after_approve_count}, operation_id={payload.get('operation_id')}, "
            f"tools={tools}, results={results!r}"
        )
        return {
            "status": "pass" if ok else "fail",
            "observed": observed,
            "error": None if ok else observed,
            "agents_used": _agents_used(trajectory),
            "tools_called": tools,
            "latency_ms": int((t1 - t0) * 1000) + int((t3 - t2) * 1000),  # виконання графа, БЕЗ очікування людини
            "human_wait_ms": int((t2 - t1) * 1000),  # скриптований approve надходить одразу — не вигадане число
        }


# --- Оркестрація --------------------------------------------------------------

_MODEL_BY_PROVIDER = {
    "demo": None,
    "openrouter": "google/gemini-3.6-flash (через OpenRouter, див. .env OPENROUTER_MODEL)",
}


async def _run_all(mode: str) -> list[EvalRecord]:
    provider = "demo" if mode == "demo" else "openrouter"
    model_name = _MODEL_BY_PROVIDER.get(provider)

    scenarios: list[tuple[str, str, str, str, CheckFn | None]] = [
        ("EVAL-01", "simple", EVAL01_QUERY, EVAL01_EXPECTED, _check_eval01),
        ("EVAL-02", "multi-step", EVAL02_QUERY, EVAL02_EXPECTED, _check_eval02),
        ("EVAL-03", "RAG-heavy", EVAL03_QUERY, EVAL03_EXPECTED, _check_eval03),
        ("EVAL-04", "cross-agent", EVAL04_QUERY, EVAL04_EXPECTED, _check_eval04),
        ("EVAL-05", "HITL", EVAL05_QUERY, EVAL05_EXPECTED, None),
    ]

    records: list[EvalRecord] = []
    for scenario_id, scenario_type, query, expected, check in scenarios:
        if provider != "demo":
            # Live-режим: якщо ключа немає, MASApp._build_llm сам підніме
            # ValueError — тут перехоплюємо ДО виклику (для БУДЬ-ЯКОГО
            # сценарію, включно з EVAL-05, де check is None), щоб позначити
            # сценарій "blocked" (не входить у знаменник pass-rate), а не
            # впасти всім прогоном чи хибно показати "fail".
            import os

            if not os.getenv("OPENROUTER_API_KEY"):
                records.append(
                    EvalRecord(
                        scenario_id=scenario_id,
                        scenario_type=scenario_type,
                        query=query,
                        status="blocked",
                        latency_ms=0,
                        agents_used=[],
                        tools_called=[],
                        mode=mode,
                        expected=expected,
                        observed="OPENROUTER_API_KEY відсутній — live-режим недоступний.",
                        error="missing OPENROUTER_API_KEY",
                        model=model_name,
                    )
                )
                continue
        try:
            if check is not None:
                outcome = await _run_query_scenario(scenario_id, query, provider, check)
            else:
                outcome = await _run_eval05(provider)
        except Exception as exc:  # непередбачений збій самого прогону (не assertion) -> явний fail, не мовчазний skip
            outcome = {
                "status": "fail",
                "observed": f"Виключення під час прогону сценарію: {exc}",
                "error": str(exc),
                "agents_used": [],
                "tools_called": [],
                "latency_ms": 0,
                "human_wait_ms": None,
            }
        records.append(
            EvalRecord(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                query=query,
                status=outcome["status"],
                latency_ms=outcome["latency_ms"],
                agents_used=outcome["agents_used"],
                tools_called=outcome["tools_called"],
                mode=mode,
                expected=expected,
                observed=outcome["observed"],
                error=outcome["error"],
                human_wait_ms=outcome["human_wait_ms"],
                model=model_name,
            )
        )
    return records


def _reset_runtime() -> None:
    """Перестворити runtime/evals і скинути fixtures перед оцінюванням."""
    if _RUNTIME_DIR.exists():
        shutil.rmtree(_RUNTIME_DIR)
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    configure_storage(_KNOWLEDGE_PATH, _BOOKING_PATH)


def _versions() -> dict[str, str]:
    from importlib.metadata import version

    return {"langchain_core": version("langchain-core"), "langgraph": version("langgraph")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="demo (за замовчуванням, без LLM-квоти) чи live (provider=openrouter, реальні LLM-виклики).",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    _reset_runtime()
    records = asyncio.run(_run_all(args.mode))
    versions = _versions()
    for record in records:
        record.versions = versions

    executed = [r for r in records if r.status in ("pass", "fail")]
    passed = [r for r in executed if r.status == "pass"]
    blocked = [r for r in records if r.status == "blocked"]
    pass_rate = (len(passed) / len(executed)) if executed else 0.0

    payload = {
        "mode": args.mode,
        "pass_rate": round(pass_rate, 4),
        "passed": len(passed),
        "executed": len(executed),
        "blocked": len(blocked),
        "results": [r.to_dict() for r in records],
    }
    _RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Режим: {args.mode}")
    for r in records:
        print(f"  {r.scenario_id} [{r.scenario_type}] -> {r.status.upper()} ({r.latency_ms} ms)")
        if r.status == "fail":
            print(f"    observed: {r.observed}")
    print(f"\nPass-rate: {len(passed)}/{len(executed)} = {pass_rate:.0%} (blocked: {len(blocked)}, не входить у знаменник)")
    print(f"Результати: {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
