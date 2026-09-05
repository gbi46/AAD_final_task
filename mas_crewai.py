"""Еквівалентна MAS у CrewAI (S14, Bonus Level, `claude_plan/05_BONUS.md`).

Той самий домен/кейс, що й `mas_langgraph.py` (S05+): billing (бюджет),
tech (погода/активності), researcher (туристичні правила/FAQ через RAG),
з coordinator-вузлом, що маршрутизує запит. Підключено до ТОГО САМОГО
`mcp_server.py` через СПРАВЖНІЙ MCP stdio subprocess
(`crewai_tools.MCPServerAdapter`) — план прямо вимагає це: "самі локальні
wrappers без MCP не задовольняють вимогу" (перевірено, що
`AAD_practical_task-2/mas_crewai.py`, попередній довідковий приклад, якраз
використовує локальні Python-обгортки, а не MCP-протокол — тому НЕ
відтворено тут, лише роль/архітектура запозичена).

Архітектура: hierarchical crew (`Process.hierarchical`) — coordinator
(`manager_agent`) делегує billing/tech/researcher-спеціалістам; це
структурний еквівалент `mas_langgraph.py`'s supervisor + 3 спеціалізовані
агенти (`general` тут не потрібен — план прямо каже "трьома тими самими
спеціалістами billing, tech, researcher і coordinator").

Per-agent tool permissions (план: "не дати всім агентам привілейований
book_hotel через спільний MCP список"; порівняй `guardrails.
TOOL_PERMISSIONS`, S08): КОЖЕН спеціаліст отримує ОКРЕМИЙ
`MCPServerAdapter(params, *allowed_tool_names)` — сам адаптер фільтрує
список tools за іменами (перевірено емпірично: `MCPServerAdapter(params,
"get_weather", "find_activities").tools` повертає РІВНО ці два tools, не
всі п'ять — див. `codex_review/evidence/` S14), тому tech фізично НЕ бачить
`calculate_trip_cost`/`book_hotel` як опцію виклику, а не лише "заборонено
post-hoc". `book_hotel` НАВМИСНО не переданий ЖОДНОМУ агенту тут: CrewAI
(на відміну від LangGraph, S09) не має вбудованого
`interrupt()`/`Command(resume=...)` для паузи графа під час kickoff, тому
чесний вибір — узагалі не давати ризиковий tool без реального HITL-
механізму, а не імітувати підтвердження, якого немає (план: "Ризикові дії
не запускати непідтвердженими; для порівняння взяти readonly scenarios").

LLM: той самий провайдер/модель, що й `mas_langgraph.py(provider=
"openrouter")` — LiteLLM-рядок `openrouter/<OPENROUTER_MODEL>` з тим самим
`OPENROUTER_API_KEY` у `.env`. Важлива відмінність від LangGraph-версії:
CrewAI не має еквіваленту `provider="demo"` — кожен `Agent` ініціалізує й
викликає РЕАЛЬНИЙ LLM (навіть побудова `Agent()` без kickoff вимагає
робочого ключа), тому запуск цього файлу ЗАВЖДИ live, ніколи не
детермінований stub.

Cycles/errors: `max_iter`/`max_execution_time` на кожному `Agent` — той
самий намір, що й `ExecutionGuard` у `mas_langgraph.py` (S05), інший API.
`allow_delegation=False` для billing/tech/researcher —
лише coordinator делегує (план: "Виконавчим агентам вимкнути непотрібне
делегування").

Ізольований venv: `crewai`/`crewai-tools` тягнуть pinned-версії, що
конфліктують з основним `requirements.txt` (понижують chromadb 1.5.9→1.1.1,
mcp 1.29.1→1.28.1, pydantic 2.13.5→2.12.5 — перевірено `pip install
--dry-run`, S14). Тому `.venv-crewai/` — ОКРЕМИЙ venv лише для цього
файлу (`.gitignore`, той самий підхід, що й основний `.venv` у S01, інший
каталог). MCP-СЕРВЕР (`mcp_server.py`) при цьому запускається дочірнім
процесом через `.venv/bin/python` (основний venv, ті самі pinned версії
chromadb/mcp, що й уся решта MAS) — `.venv-crewai` містить лише
CrewAI-клієнта, що говорить MCP-протоколом (сумісний між мінорними
версіями `mcp`), а не бізнес-логіку сервера.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent

# Per-agent allowlist -- дзеркалить guardrails.TOOL_PERMISSIONS (S08) для
# LangGraph-версії; book_hotel НАВМИСНО відсутній у жодному записі (див.
# docstring модуля).
TECH_TOOLS = ("get_weather", "find_activities")
BILLING_TOOLS = ("calculate_trip_cost", "search_knowledge")
RESEARCHER_TOOLS = ("search_knowledge",)

_DEFAULT_MODEL_SLUG = "anthropic/claude-sonnet-4.6"
_MAX_ITER = 6  # той самий порядок величини, що й ExecutionGuard.max_steps (S05)
# 120 c, не 60: coordinator синхронно чекає на ВСЮ делеговану роботу
# спеціаліста (кілька tool-раундів через реальний MCP subprocess) --
# перший live-прогін researcher-сценарію ("Які правила відвідування
# Лувру?") впав у TimeoutError за 60с, бо live-модель (Claude Sonnet 4.6)
# зробила 7+ повторних search_knowledge-запитів із трохи різними
# формулюваннями, не задовольнившись коротким статичним корпусом (10
# документів) з першої спроби -- виявлено емпірично, задокументовано як
# відомий edge case нижче, а не приховано.
_MAX_EXECUTION_TIME = 120  # секунд на агента -- аналог ExecutionGuard.timeout_seconds


def _prepare_crewai_storage() -> None:
    """CrewAI сам пише у ~/.local/share/crewai/... за замовчуванням -- тримати
    це у runtime/ репозиторію (gitignored), не в $HOME користувача (той
    самий підхід, що в AAD_practical_task-2/mas_crewai.py)."""
    storage_dir = _ROOT / "runtime" / "crewai_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CREWAI_STORAGE_DIR", str(storage_dir))


def _build_llm() -> Any:
    """`LLM(...)`, не голий рядок -- CrewAI/OpenRouter за замовчуванням
    підставляють дуже високий `max_tokens` (65536), що перевищує баланс
    тестового ключа (те саме, що й у `mas_langgraph.MASApp._build_llm` для
    ChatOpenAI/OpenRouter, S05/pre-S10) -- `openai.APIStatusError: 402`
    без явного ліміту."""
    from crewai import LLM

    slug = os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL_SLUG)
    max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))
    return LLM(model=f"openrouter/{slug}", max_tokens=max_tokens)


def _mcp_server_params(runtime_dir: Path) -> Any:
    """StdioServerParameters -- команда явно вказує на ГОЛОВНИЙ .venv
    (не .venv-crewai, у якому запущено сам цей файл): mcp_server.py
    залежить від chromadb/mcp/pydantic з їхніми ТОЧНИМИ pinned-версіями
    основного requirements.txt (S01), а не від того, що встановлено для
    CrewAI-клієнта."""
    from mcp import StdioServerParameters

    main_python = str(_ROOT / ".venv" / "bin" / "python")
    server_script = str(_ROOT / "mcp_server.py")
    return StdioServerParameters(
        command=main_python,
        args=[server_script],
        env={
            **os.environ,
            "MCP_KNOWLEDGE_PATH": str(runtime_dir / "chroma_db"),
            "MCP_BOOKING_PATH": str(runtime_dir / "bookings.json"),
        },
    )


@contextlib.contextmanager
def _connected_tools(runtime_dir: Path):
    """Відкриває ТРИ окремі MCP subprocess-и (по одному на спеціаліста,
    кожен зі своїм відфільтрованим списком tools через
    `MCPServerAdapter(params, *tool_names)`) і повертає dict
    {agent_key: [crewai Tool, ...]}. Усі три вказують на ОДНАКОВІ
    knowledge/booking шляхи (runtime_dir) -- окремі процеси, той самий
    persisted стан.
    """
    from crewai_tools import MCPServerAdapter

    params = _mcp_server_params(runtime_dir)
    with contextlib.ExitStack() as stack:
        tech_tools = stack.enter_context(MCPServerAdapter(params, *TECH_TOOLS))
        billing_tools = stack.enter_context(MCPServerAdapter(params, *BILLING_TOOLS))
        researcher_tools = stack.enter_context(MCPServerAdapter(params, *RESEARCHER_TOOLS))
        yield {
            "tech": list(tech_tools),
            "billing": list(billing_tools),
            "researcher": list(researcher_tools),
        }


def build_crew(runtime_dir: Path | None = None, tools_by_agent: dict[str, list[Any]] | None = None):
    """Створює CrewAI hierarchical crew.

    `tools_by_agent` дозволяє передати вже підключені (через
    `_connected_tools`) MCP-tools ззовні -- потрібно, бо `MCPServerAdapter`
    керує субпроцесом через контекстний менеджер (`with`), а `Crew.kickoff()`
    має виконатись УСЕРЕДИНІ цього `with`-блоку (інакше subprocess-и
    закриються раніше, ніж агенти встигнуть їх викликати).
    """
    _prepare_crewai_storage()
    runtime_dir = runtime_dir or (_ROOT / "runtime" / "crewai")
    runtime_dir.mkdir(parents=True, exist_ok=True)

    from crewai import Agent, Crew, Process, Task

    model = _build_llm()
    tools_by_agent = tools_by_agent or {"tech": [], "billing": [], "researcher": []}

    coordinator = Agent(
        role="Travel Coordinator",
        goal="Визначити тип туристичного запиту і делегувати правильному спеціалісту",
        backstory=(
            "Ти координатор туристичної підтримки. Визначаєш, чи запит про "
            "бюджет/вартість (billing), погоду/активності/маршрут (tech), чи "
            "туристичні правила/FAQ (researcher), і делегуєш саме туди. Сам "
            "жодних обчислень чи пошуку не виконуєш."
        ),
        llm=model,
        allow_delegation=True,
        max_iter=_MAX_ITER,
        max_execution_time=_MAX_EXECUTION_TIME,
        verbose=True,
    )

    billing_agent = Agent(
        role="Billing Specialist",
        goal="Оцінити орієнтовний бюджет подорожі та відповісти на запити про вартість",
        backstory=(
            "Ти спеціаліст із бюджету подорожей. Використовуєш "
            "calculate_trip_cost для точних розрахунків і search_knowledge "
            "для питань про політику бронювання. Ти НЕ виконуєш жодних "
            "бронювань (book_hotel тобі недоступний навіть якщо про це "
            "просять) — лише оцінка вартості."
        ),
        llm=model,
        tools=tools_by_agent["billing"],
        allow_delegation=False,
        max_iter=_MAX_ITER,
        max_execution_time=_MAX_EXECUTION_TIME,
        verbose=True,
    )

    tech_agent = Agent(
        role="Tech & Logistics Specialist",
        goal="Надати прогноз погоди та підібрати активності за інтересом",
        backstory=(
            "Ти спеціаліст з погоди, маршруту й активностей. Використовуєш "
            "get_weather і find_activities. Питання про бюджет чи "
            "туристичні правила передаєш координатору для делегування."
        ),
        llm=model,
        tools=tools_by_agent["tech"],
        allow_delegation=False,
        max_iter=_MAX_ITER,
        max_execution_time=_MAX_EXECUTION_TIME,
        verbose=True,
    )

    researcher_agent = Agent(
        role="Travel Rules Researcher",
        goal="Відповісти на питання про туристичні правила, візові вимоги, музеї та безпеку",
        backstory=(
            "Ти дослідник туристичних правил. Використовуєш search_knowledge "
            "(RAG над базою travel-01..10) і завжди цитуєш ID документа у "
            "відповіді. Питання про бюджет чи погоду передаєш координатору."
        ),
        llm=model,
        tools=tools_by_agent["researcher"],
        allow_delegation=False,
        max_iter=_MAX_ITER,
        max_execution_time=_MAX_EXECUTION_TIME,
        verbose=True,
    )

    task = Task(
        description=(
            "Оброби туристичний запит користувача: {query}\n"
            "1. Визнач, чи це billing (бюджет/вартість), tech (погода/"
            "активності/маршрут) чи researcher (правила/FAQ/RAG).\n"
            "2. Делегуй ЛИШЕ ОДНОМУ найбільш підходящому спеціалісту "
            "(або двом послідовно, якщо запит справді стосується і "
            "погоди, і бюджету).\n"
            "3. Поверни фінальну відповідь користувачу українською мовою."
        ),
        expected_output="Коротка, конкретна відповідь українською мовою на основі реальних даних інструментів.",
        # `agent=` НЕ вказується навмисно: `Crew._update_manager_tools`
        # (crewai 1.15.20) будує delegation tools з `[task.agent]`, коли
        # `task.agent` заданий, замість `self.agents` (billing/tech/
        # researcher) -- тобто явний `agent=coordinator` тут ЛАМАЄ
        # делегування (менеджер бачить лише самого себе як "коворкера" і
        # отримує "Executor is already running" при спробі делегувати).
        # Виявлено емпірично під час першого live-прогону S14 (codex_review
        # evidence). Без `agent=` `Process.hierarchical` сам призначає
        # `manager_agent` виконавцем задачі й коректно пропонує ЙОМУ
        # делегувати на `self.agents`.
    )

    return Crew(
        agents=[billing_agent, tech_agent, researcher_agent],
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=coordinator,
        verbose=True,
    )


def run_query(query: str, runtime_dir: Path | None = None) -> str:
    """Точка входу для одного запиту -- відкриває MCP-підключення, будує
    crew, виконує kickoff, коректно закриває subprocess-и навіть при
    винятку (`with` навколо всього виклику)."""
    runtime_dir = runtime_dir or (_ROOT / "runtime" / "crewai")
    with _connected_tools(runtime_dir) as tools_by_agent:
        crew = build_crew(runtime_dir, tools_by_agent)
        result = crew.kickoff(inputs={"query": query})
    return str(result)


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY відсутній у .env -- mas_crewai.py вимагає live LLM (немає demo-режиму).")
        raise SystemExit(1)

    query = sys.argv[1] if len(sys.argv) > 1 else "Скільки коштуватиме подорож до Львова на 3 дні для двох?"
    print(f"=== {query} ===")
    print(run_query(query))
