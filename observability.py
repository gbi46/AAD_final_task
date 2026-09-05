"""Observability для MAS через LangSmith (S10).

Налаштування — виключно змінними середовища. LangChain/LangGraph самі
трасують кожен `.invoke()`/`.ainvoke()` на `Runnable` (LLM-виклики,
структуровані виходи, tool calls, вкладені підграфи), щойно
`LANGCHAIN_TRACING_V2=true` і валідний ключ заданий — жодного додаткового
коду для базової ієрархії run'ів не потрібно.

Ієрархія trace для одного запиту (root run → ... → handoff) відповідає
структурі самого MAS-графа (S05/S06):

    handle_query (root run, thread_id/run_id у метаданих)
    └─ supervisor              (RouteDecision: with_structured_output)
    └─ billing / tech / researcher / general   (specialist)
        └─ billing: planner → executor → replanner (вкладений підграф,
           executor → dispatch_tool → tool/MCP call)
        └─ tech: agent → tools (ReAct-цикл)
        └─ ... (handoff tech→billing: другий візит supervisor +
           specialist-вузол у ТОМУ САМОМУ root run/thread_id — видно як
           повторний supervisor-run з `pending_approval`/`handoff_context`
           у вхідному стані)

Чутливі дані:
- Секрети (API-ключі з .env) НІКОЛИ не потрапляють у жоден prompt чи
  аргумент tool — LangSmith їх і не побачить, бо трасується лише те, що
  реально проходить через LangChain `Runnable.invoke`.
- Синтетичні PII (потрібні для red-team перевірок, S12) — ЧЕСНЕ
  застереження: `output_guardrail` (S08) маскує PII в `results`/
  `messages` НА РІВНІ agent-вузла графа (`MASApp._apply_output_guardrail`)
  і в `trajectory.json` (`trajectory_logger._redact_text`) — ОБИДВА після
  того, як сам LLM-виклик уже стався. LangSmith же трасує сирі
  prompt/response LangChain `Runnable` НИЖЧЕ цього рівня (безпосередньо
  callback з LLM-клієнта) — тобто наш output_guardrail НЕ встигає
  редагувати те, що LangSmith встигає захопити до нашого вузла. Повний
  захист вимагав би кастомного LangChain callback/masking hook на рівні
  самого tracer'а, якого тут немає. Тому: демонстраційні live-запити для
  S10 навмисно НЕ містять реальних чи синтетичних PII (лише туристичні
  дані — міста, дати, суми); red-team сценарії з PII (RT-02) запускаються
  БЕЗ активного LangSmith tracing, щоб синтетичні тестові PII не
  потрапляли до стороннього сервісу навіть у масктестовому вигляді. Ця
  залишкова прогалина задокументована чесно (не приховується) — S13.
"""

from __future__ import annotations

import os
import sys


def is_tracing_enabled() -> bool:
    """Чи увімкнено (і чи є ключ) — без побічних ефектів на середовище."""
    return bool(os.getenv("LANGSMITH_API_KEY")) and os.getenv("LANGSMITH_TRACING", "false").lower() == "true"


def configure_observability(project: str | None = None) -> bool:
    """Налаштовує LangSmith tracing з environment.

    Без `LANGSMITH_API_KEY` трасування ЯВНО вимикається (не мовчазний
    no-op SDK за замовчуванням) — попередження у stderr, решта MAS працює
    як завжди (жодної залежності функціональності від observability).

    Returns:
        True, якщо трасування реально увімкнено.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print(
            "observability: LANGSMITH_API_KEY відсутній — LangSmith tracing ВИМКНЕНО. "
            "MAS працює без змін, лише без відправки traces.",
            file=sys.stderr,
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project or os.getenv("LANGSMITH_PROJECT", "aad-final-mas")
    return True


def project_url() -> str | None:
    """URL проєкту в LangSmith UI (для README/логів), якщо tracing увімкнено."""
    if not is_tracing_enabled():
        return None
    project = os.getenv("LANGSMITH_PROJECT", "aad-final-mas")
    return f"https://smith.langchain.com/projects?tab=0&name={project}"


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv(".env")
    enabled = configure_observability()
    print(f"LangSmith tracing enabled: {enabled}")
    if enabled:
        print(f"Project URL: {project_url()}")
    else:
        print("Задайте LANGSMITH_API_KEY у .env, щоб увімкнути live-трасування.")
        raise SystemExit(0)

    from tools_legacy import configure_storage

    import mas_langgraph as m

    runtime_dir = __import__("pathlib").Path(__file__).resolve().parent / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    configure_storage(runtime_dir / "chroma_db", runtime_dir / "bookings.json")

    async def main() -> None:
        mas = m.MASApp(provider="openrouter", run_id="observability-demo")
        # ≥2 live-запити, один з реальним handoff (план S10) — жодних PII.
        queries = [
            ("Скільки коштуватиме подорож до Парижа на 3 дні для двох?", "obs-billing"),
            ("Яка погода у Львові і скільки коштуватиме подорож на 3 дні?", "obs-handoff"),
        ]
        for query, thread_id in queries:
            print(f"\n=== {query} (thread_id={thread_id}) ===")
            final = await mas.handle_query(query, thread_id=thread_id)
            print("current_agent:", final.get("current_agent"))
            print("handoff_count:", final.get("handoff_count"))
            for r in final.get("results", []):
                print("result:", r[:200])

    asyncio.run(main())
    print(f"\nПереглянути traces: {project_url()}")
