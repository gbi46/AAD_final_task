"""HITL (Human-in-the-Loop) для ризикового MCP-tool `book_hotel` (S09).

`RISKY_TOOLS` тут — єдине джерело істини для того, які tools вимагають
людського підтвердження перед виконанням; `approval_gate`-вузол графа
(`MASApp._approval_gate_node` у `mas_langgraph.py`) і `tool_guardrail`
(`guardrails.py`, S08 — allowlist per agent) працюють НЕЗАЛЕЖНО одне від
одного: allowlist каже "чи агенту взагалі дозволено цей tool", HITL каже
"навіть якщо дозволено — виконати можна лише після підтвердження людини".

Механізм (деталі архітектури — у `mas_langgraph.py`):

1. `BillingAgent._executor_node`, зустрівши крок з tool із `RISKY_TOOLS`,
   НЕ виконує його (жодного dispatch) — повертає `pending_risky_step` із
   billing-підграфа назовні.
2. `MASApp._billing_node` перетворює це на `pending_approval` у
   зовнішньому стані (`agent_name`, `tool`, повні `args`, `consequences`,
   детермінований `operation_id` — той самий ключ, що й ідемпотентність
   `book_hotel_core`) і залишає `completed=False`.
3. `MASApp._supervisor_node` бачить `pending_approval` і маршрутизує
   ПРЯМО до `approval_gate` (без повторної LLM-класифікації).
4. `MASApp._approval_gate_node` викликає `interrupt(payload)` — граф
   ставиться на паузу; ніякого tool call ще не сталося. Рішення надходить
   через `Command(resume={"action": "approve"|"reject"|"edit", ...})` для
   ТОГО САМОГО `thread_id` (потребує checkpointer — `MASApp(checkpointer=...)`,
   той самий файловий `AsyncSqliteSaver`, що й у S07).

Код ДО `interrupt()` (кроки 1–3, а також початок `_approval_gate_node` до
самого виклику `interrupt()`) не має side effects — LangGraph повторно
виконує цей шлях при resume, і жоден book_hotel виклик там не станеться,
доки не отримано explicit рішення. `book_hotel_core`'s ідемпотентність
(S02) додатково захищає від дубля навіть якщо approve сталося двічі
(рестарт/replay) — той самий `operation_id`/`idempotency_key` повертає
вже наявний запис, а не створює новий.

Boolean `approved=True`, який LLM міг би сам згенерувати як частину
tool-виклику, НЕ приймається як підтвердження: `_approval_gate_node`
розпізнає лише dict-рішення з `action` рівно одним із
`{"approve", "reject", "edit"}` (перевірка нижче, `is_valid_decision`).

Демонстрація трьох сценаріїв (approve/reject/edit) — запуском цього
файлу як CLI (кожен виклик — окремий процес, checkpoint у файловому
SqliteSaver переживає його завершення, як і в S07):

    python hitl.py approve
    python hitl.py reject
    python hitl.py edit
    python hitl.py invalid-edit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from tools_legacy import configure_storage

# MASApp імпортується ліниво (у run_scenario), а не на рівні модуля: mas_langgraph.py
# сам імпортує RISKY_TOOLS/is_valid_decision звідси — top-level import тут
# спричинив би циклічний імпорт.

RISKY_TOOLS = {"book_hotel"}


def is_valid_decision(decision: Any) -> bool:
    """Дійсне людське рішення — dict з action у дозволеному переліку.

    НЕ приймає bare `True`/`{"approved": True}` — саме така форма могла б
    бути передана LLM самостійно (без реального втручання людини), а план
    прямо забороняє трактувати це як підтвердження.
    """
    return isinstance(decision, dict) and decision.get("action") in ("approve", "reject", "edit")


# --- Демонстраційний CLI: три сценарії через окремі процеси -----------------

_ROOT = Path(__file__).resolve().parent
_RUNTIME_DIR = Path(os.environ.get("HITL_RUNTIME_DIR", str(_ROOT / "runtime")))
DB_PATH = Path(os.environ.get("HITL_DB_PATH", str(_RUNTIME_DIR / "hitl_demo.db")))
KNOWLEDGE_PATH = Path(os.environ.get("HITL_KNOWLEDGE_PATH", str(_RUNTIME_DIR / "hitl_chroma_db")))
BOOKING_PATH = Path(os.environ.get("HITL_BOOKING_PATH", str(_RUNTIME_DIR / "hitl_bookings.json")))

QUERY = "Забронюй готель на 15 вересня 2026 на 2 ночі"

_RESUME_PAYLOADS: dict[str, dict[str, Any]] = {
    "approve": {"action": "approve"},
    "reject": {"action": "reject", "reason": "Клієнт скасував поїздку."},
    "edit": {"action": "edit", "args": {"nights": 3, "total_cost": 420.0}},
    "invalid-edit": {"action": "edit", "args": {"nights": -1}},  # порушує BookingInput (1..30)
}


def _booking_count() -> int:
    if not BOOKING_PATH.exists():
        return 0
    return len(json.loads(BOOKING_PATH.read_text(encoding="utf-8")))


async def run_scenario(scenario: str) -> None:
    from mas_langgraph import MASApp  # ліниво: розриває циклічний імпорт з mas_langgraph.py

    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    configure_storage(KNOWLEDGE_PATH, BOOKING_PATH)
    thread_id = f"hitl-demo-{scenario}"

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        mas = MASApp(provider="demo", run_id=f"hitl-{scenario}", checkpointer=saver)

        before_count = _booking_count()
        print(f"Бронювань ДО ({scenario}): {before_count}")

        paused_state = await mas.handle_query(QUERY, thread_id=thread_id)
        interrupts = paused_state.get("__interrupt__")
        assert interrupts, "Граф мав зупинитися на interrupt() — жодного tool call ще не відбулося"
        payload = interrupts[0].value
        print("Interrupt payload (жодної дії ще не виконано):")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        assert _booking_count() == before_count, "До рішення людини booking count не мав змінитися"

        resume_payload = _RESUME_PAYLOADS[scenario]
        final_state = await mas.app.ainvoke(Command(resume=resume_payload), config=mas.config(thread_id))
        print(f"\nРезультат ({scenario}):", final_state["results"])

        after_count = _booking_count()
        print(f"Бронювань ПІСЛЯ ({scenario}): {after_count}")

        if scenario in ("approve", "edit"):
            # edit ЗМІНЮЄ параметри (nights/total_cost) перед виконанням —
            # це інший idempotency_key, тому це НОВЕ (не дублююче)
            # бронювання, а не той самий запис, що й approve.
            assert after_count == before_count + 1, f"{scenario} мав створити рівно один booking"
        else:
            assert after_count == before_count, f"{scenario} НЕ мав створювати booking side effect"


_CROSS_PROCESS_THREAD_ID = "hitl-demo-cross-process"


async def run_pause_only() -> None:
    """Фаза 1 (окремий процес): дійти до interrupt() і завершитись —
    жодного tool call ще не відбулося. Демонструє "пауза → новий процес →
    resume" по-справжньому (а не в межах одного інтерпретатора, як
    `run_scenario`)."""
    from mas_langgraph import MASApp

    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    configure_storage(KNOWLEDGE_PATH, BOOKING_PATH)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        mas = MASApp(provider="demo", run_id="hitl-pause", checkpointer=saver)
        before_count = _booking_count()
        print(f"Бронювань ДО паузи: {before_count}")
        paused_state = await mas.handle_query(QUERY, thread_id=_CROSS_PROCESS_THREAD_ID)
        interrupts = paused_state.get("__interrupt__")
        assert interrupts, "Граф мав зупинитися на interrupt()"
        print("Interrupt payload:")
        print(json.dumps(interrupts[0].value, ensure_ascii=False, indent=2))
        assert _booking_count() == before_count, "До рішення людини booking count не мав змінитися"
    print("Процес (pause) завершено: з'єднання закрито, жодного tool call не відбулося.")


async def run_resume_only(action: str) -> None:
    """Фаза 2 (НОВИЙ процес): нове з'єднання до того самого DB/thread_id,
    відновлення через Command(resume=...)."""
    from mas_langgraph import MASApp

    configure_storage(KNOWLEDGE_PATH, BOOKING_PATH)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        mas = MASApp(provider="demo", run_id="hitl-resume", checkpointer=saver)
        before_count = _booking_count()
        final_state = await mas.app.ainvoke(
            Command(resume=_RESUME_PAYLOADS[action]), config=mas.config(_CROSS_PROCESS_THREAD_ID)
        )
        print("Результат:", final_state["results"])
        after_count = _booking_count()
        print(f"Бронювань: {before_count} -> {after_count}")

        # Повторний resume (той самий thread_id) — граф уже completed,
        # LangGraph просто повертає останній стан без жодного побічного
        # ефекту (немає interrupt, на який можна було б resume вдруге).
        replay_state = await mas.app.ainvoke(None, config=mas.config(_CROSS_PROCESS_THREAD_ID))
        replay_count = _booking_count()
        print(f"Повторний resume (replay) — бронювань: {after_count} -> {replay_count} (без дублю)")
        assert replay_count == after_count, "Повторний resume не мав створити ще один booking"
        assert replay_state["results"] == final_state["results"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in sorted(_RESUME_PAYLOADS):
        subparsers.add_parser(name, help=f"Сценарій '{name}': пауза+resume в ОДНОМУ процесі (швидка демонстрація).")

    subparsers.add_parser("pause", help="Лише дійти до interrupt() і завершити процес (фаза 1 cross-process демо).")
    resume_parser = subparsers.add_parser(
        "resume", help="Новий процес: відновити з того самого checkpoint (фаза 2 cross-process демо)."
    )
    resume_parser.add_argument("action", choices=sorted(_RESUME_PAYLOADS))

    args = parser.parse_args()
    if args.command == "pause":
        asyncio.run(run_pause_only())
    elif args.command == "resume":
        asyncio.run(run_resume_only(args.action))
    else:
        asyncio.run(run_scenario(args.command))


if __name__ == "__main__":
    main()
