"""Демонстрація durable state MAS через файловий SqliteSaver (S07).

Запуск — ДВА ОКРЕМІ процеси (не два виклики в одному інтерпретаторі; саме
так план вимагає довести, що стан реально пережив закриття процесу, а не
просто лишився в пам'яті одного Python-об'єкта):

    python persistence_demo.py pause    # процес А
    python persistence_demo.py resume   # процес Б (новий інтерпретатор)

Процес А: створює billing-запит, виконує рівно ОДИН крок графа (supervisor
— маршрутизація), ставить граф на паузу (`interrupt_after=["supervisor"]`),
знімає snapshot стану, закриває з'єднання виходом з `async with` (як
завершення процесу) — жодного tool call ще не відбулося.

Процес Б: відкриває НОВЕ з'єднання до того самого файлу `agent_state.db` і
того самого `thread_id`, зчитує збережений checkpoint, продовжує граф з
місця паузи (`resume()` — `None` як вхід) до завершення billing-підграфа.
Порівнює `plan`/`current_step`/`results` до і після — доводить, що
executor виконався РІВНО ОДИН раз (не повторно після відновлення), а не
лише що фінальний результат виглядає правильно.

Обидва процеси також перевіряють ізоляцію: інший `thread_id`
(`OTHER_THREAD_ID`) не бачить жодного стану першого.

Async, не sync `SqliteSaver`: граф викликається через `ainvoke`
(dispatch_tool — async), а синхронний `SqliteSaver` явно не підтримує
async-методи (LangGraph одразу підіймає `NotImplementedError` з
підказкою використати `AsyncSqliteSaver` — перевірено емпірично).

Після фази `resume`, ПІСЛЯ закриття `AsyncSqliteSaver` (окремим коротким
sync-з'єднанням), SQLite явно checkpoint'иться
(`PRAGMA wal_checkpoint(TRUNCATE)`) і повертається у `journal_mode=DELETE`
— `SqliteSaver`/`AsyncSqliteSaver` вмикають WAL (`PRAGMA journal_mode=WAL`),
а WAL-режим тримає недавні записи в окремому файлі `-wal` до checkpoint;
без цього кроку доставлений `agent_state.db` був би неповним знімком без
супутнього `-wal` файлу. Результат — самодостатній `agent_state.db` у
корені репозиторію (контрольний артефакт здачі, S07), без потреби у
відсутньому WAL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from sqlite3 import connect

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mas_langgraph import MASApp
from tools_legacy import configure_storage
from trajectory_logger import save_trajectories

_ROOT = Path(__file__).resolve().parent

# Шляхи перевизначаються через env (тести спрямовують їх у tempdir, щоб не
# чіпати справжні контрольні артефакти здачі в корені репозиторію) — типово
# ж це саме контрольні артефакти S07: agent_state.db, chroma_db/, trajectory.json.
DB_PATH = Path(os.environ.get("PERSISTENCE_DB_PATH", str(_ROOT / "agent_state.db")))
KNOWLEDGE_PATH = Path(os.environ.get("PERSISTENCE_KNOWLEDGE_PATH", str(_ROOT / "chroma_db")))
TRAJECTORY_PATH = Path(os.environ.get("PERSISTENCE_TRAJECTORY_PATH", str(_ROOT / "trajectory.json")))
_RUNTIME_DIR = Path(os.environ.get("PERSISTENCE_RUNTIME_DIR", str(_ROOT / "runtime")))
BOOKING_PATH = _RUNTIME_DIR / "bookings.json"  # не потрібен цим сценарієм (book_hotel не викликається)
_PAUSE_TRAJECTORY_TMP = _RUNTIME_DIR / "trajectory_persistence_pause.json"

THREAD_ID = "persistence-demo-billing"
OTHER_THREAD_ID = "persistence-demo-isolated"
QUERY = "Скільки коштуватиме подорож до Парижа на 5 днів для двох?"

_EMPTY_SNAPSHOT = {"plan": [], "current_step": 0, "results": [], "completed": False}


async def _snapshot(mas: MASApp, thread_id: str) -> dict:
    values = await mas.get_snapshot(thread_id)
    return {
        "plan": values.get("plan", []),
        "current_step": values.get("current_step", 0),
        "results": values.get("results", []),
        "completed": values.get("completed", False),
    }


async def run_pause() -> None:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    configure_storage(KNOWLEDGE_PATH, BOOKING_PATH)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        # interrupt_after=["supervisor"]: пауза одразу після ОДНОГО виконаного
        # кроку (маршрутизація), ще до того, як billing-підграф реально
        # виконає жоден tool call.
        mas = MASApp(
            provider="demo", run_id="persistence-demo-pause", checkpointer=saver, interrupt_after=["supervisor"]
        )

        await mas.handle_query(QUERY, thread_id=THREAD_ID)
        before = await _snapshot(mas, THREAD_ID)
        print("Знімок ПІСЛЯ паузи (один крок supervisor, billing ще НЕ виконувався):")
        print(json.dumps(before, ensure_ascii=False, indent=2))
        assert before["current_step"] == 0, "billing executor не мав виконатись до паузи"
        assert before["results"] == [], "results мають бути порожніми до паузи"

        other = await _snapshot(mas, OTHER_THREAD_ID)
        print("\nЗнімок ІНШОГО thread_id (має бути порожнім — ізоляція сесій):")
        print(json.dumps(other, ensure_ascii=False, indent=2))
        assert other == _EMPTY_SNAPSHOT

        save_trajectories([mas.logger.as_dict()], _PAUSE_TRAJECTORY_TMP)
    # Вихід з `async with` закриває aiosqlite-з'єднання — імітація завершення процесу А.
    print(f"\nПроцес А (pause) завершено: з'єднання {DB_PATH} закрито.")


async def run_resume() -> None:
    if not DB_PATH.exists():
        print("Спочатку запустіть: python persistence_demo.py pause", file=sys.stderr)
        raise SystemExit(1)

    configure_storage(KNOWLEDGE_PATH, BOOKING_PATH)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:  # НОВЕ з'єднання, той самий файл
        # Без interrupt_after: цей граф доводить billing до кінця.
        mas = MASApp(provider="demo", run_id="persistence-demo-resume", checkpointer=saver)

        before = await _snapshot(mas, THREAD_ID)
        print("Стан ДО продовження (зчитано з нового з'єднання, новий процес):")
        print(json.dumps(before, ensure_ascii=False, indent=2))

        await mas.resume(THREAD_ID)
        after = await _snapshot(mas, THREAD_ID)
        print("\nСтан ПІСЛЯ продовження (billing довиконано до кінця):")
        print(json.dumps(after, ensure_ascii=False, indent=2))

        assert after["completed"] is True, "billing має завершитись після resume"
        assert len(after["plan"]) >= 1
        # Відсутність повторного side effect: рівно один результат на крок
        # плану (НЕ вдвічі більше — це означало б, що executor виконався
        # і до, і після паузи).
        assert len(after["results"]) == len(after["plan"]), (
            f"results мають відповідати кількості кроків плану РІВНО один раз: "
            f"{len(after['results'])} результатів на {len(after['plan'])} кроків"
        )
        print(f"\nplan: {before['plan']} -> {after['plan']}")
        print(f"current_step: {before['current_step']} -> {after['current_step']}")
        print("results (кожен рядок — ОДИН виконаний крок, без повторів):")
        for line in after["results"]:
            print(f"  - {line}")

        other = await _snapshot(mas, OTHER_THREAD_ID)
        print("\nІнший thread_id і зараз не бачить стану першого (перевірено повторно, з нового процесу):")
        print(json.dumps(other, ensure_ascii=False, indent=2))
        assert other == _EMPTY_SNAPSHOT

        # Контрольний артефакт здачі: об'єднати обидві фази у trajectory.json.
        runs = []
        if _PAUSE_TRAJECTORY_TMP.exists():
            runs.extend(json.loads(_PAUSE_TRAJECTORY_TMP.read_text(encoding="utf-8"))["runs"])
        runs.append(mas.logger.as_dict())
        save_trajectories(runs, TRAJECTORY_PATH)
        print(f"\ntrajectory.json (обидві фази, {len(runs)} runs) збережено у {TRAJECTORY_PATH}.")
    # Вихід з `async with` закриває aiosqlite-з'єднання.

    # Окреме коротке sync-з'єднання лише для WAL-checkpoint: доставлений
    # agent_state.db має бути самодостатнім знімком без потреби у -wal.
    connection = connect(str(DB_PATH), check_same_thread=False)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    connection.execute("PRAGMA journal_mode=DELETE;")
    connection.commit()
    connection.close()
    wal_path = DB_PATH.with_name(DB_PATH.name + "-wal")
    shm_path = DB_PATH.with_name(DB_PATH.name + "-shm")
    print(
        f"\nПроцес Б (resume) завершено: {DB_PATH} checkpoint'нуто "
        f"(WAL sidecar лишився: {wal_path.exists()}, {shm_path.exists()})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pause", "resume"])
    args = parser.parse_args()
    if args.phase == "pause":
        asyncio.run(run_pause())
    else:
        asyncio.run(run_resume())


if __name__ == "__main__":
    main()
