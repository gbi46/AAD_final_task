"""Мультиагентна система (MAS) туристичної координації над LangGraph (S05+S06).

Supervisor-вузол маршрутизує запит між чотирма спеціалізованими агентами:

- `billing` — бюджет і бронювання: підграф planner → executor → replanner,
  перенесений і адаптований з `Kravchuk_AAD_2/plan_execute.py` (без
  persistence/HITL — це S07/S09; `book_hotel` тут НЕ викликається
  автоматично, лише `calculate_trip_cost`/`search_knowledge`).
- `tech` — погода, маршрут, активності: ReAct LLM → tools → LLM,
  перенесений і адаптований з `Kravchuk_AAD_1/agent.py`.
- `researcher` — туристичні правила й FAQ: LLM вирішує, чи потрібен
  `search_knowledge` (Agentic RAG), відповідь цитує ID документів; при
  підключеному MCP спершу перевіряє статичний resource `faq://general`
  (дешевше за RAG) і лише тоді звертається до tool.
- `general` — привітання й уточнення без ризикових викликів; при
  підключеному MCP формує відповідь через prompt `travel_reply`.

Усі три агенти викликають tools через ОДИН спільний dispatcher
(`MASApp.dispatch_tool`, S06) — той самий метод обслуговує локальні
LangChain-обгортки з `tools_legacy.py` (за замовчуванням) і MCP-tools,
завантажені через `MultiServerMCPClient` після `await MASApp.connect_mcp()`
(stdio, абсолютний шлях до `mcp_server.py`, поточний `sys.executable`). Це
та сама точка, куди S08 додасть guardrail-політики (allowlist per agent,
rate-limit).
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from guardrails import RateLimiter, input_guardrail, output_guardrail, tool_guardrail
from safety import ExecutionGuard, SafetyStop, call_with_timeout
from hitl import RISKY_TOOLS, is_valid_decision
from tools_legacy import TOOLS_BY_NAME, BookingInput, book_hotel_operation_id
from trajectory_logger import TrajectoryLogger

AgentName = Literal["billing", "tech", "researcher", "general", "approval_gate"]

MAX_HANDOFFS = 1
MAX_SUPERVISOR_VISITS = 8


class ToolDispatchError(RuntimeError):
    """Підняте dispatcher'ом (`MASApp.dispatch_tool`), коли виклик tool
    завершився помилкою (невідомий інструмент, збій MCP-сервера тощо).
    Агенти ловлять це виключення і формують явне повідомлення про помилку —
    збій НІКОЛИ не повертається як фальшивий успішний результат.
    """


# --- Спільні допоміжні функції (місто/дні/мандрівники з тексту запиту) -----

_CITY_ALIASES = {
    "києв": "Київ",
    "київ": "Київ",
    "львов": "Львів",
    "львів": "Львів",
    "одес": "Одеса",
    "варшав": "Варшава",
    "берлін": "Берлін",
    "париж": "Париж",
}


def _extract_city(text: str, default: str = "Київ") -> str:
    lowered = text.lower()
    return next((city for stem, city in _CITY_ALIASES.items() if stem in lowered), default)


def _extract_days(text: str, default: int = 1) -> int:
    match = re.search(r"(\d+)\s*дн", text)
    return int(match.group(1)) if match else default


def _extract_travelers(text: str) -> int:
    lowered = text.lower()
    return 2 if any(marker in lowered for marker in ("двох", "2 ос", "для 2")) else 1


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _stringify_tool_result(result: Any) -> str:
    """Нормалізує результат tool до єдиного текстового рядка.

    Локальні LangChain-обгортки з `tools_legacy.py` повертають готовий
    `str`. MCP-tool через `.ainvoke()` повертає список content-блоків
    (`[{"type": "text", "text": "...", ...}, ...]`) — без цієї нормалізації
    агенти отримували б Python-репр списку словників замість тексту.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(result)


# --- Стан MAS і reducers -----------------------------------------------------


def _merge_trajectory(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Додає нові кроки траєкторії без дублювання (ключ — agent_name+node_name+step_number,
    step_number монотонно зростає в межах одного спільного TrajectoryLogger).
    """
    seen = {(item.get("agent_name"), item.get("node_name"), item.get("step_number")) for item in existing}
    merged = list(existing)
    for item in new:
        key = (item.get("agent_name"), item.get("node_name"), item.get("step_number"))
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


class MASState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: AgentName
    plan: list[str]  # явна заміна (без reducer) — актуально для billing-підграфа
    current_step: int
    results: Annotated[list[str], operator.add]
    step_count: int  # глобальний лічильник візитів supervisor
    trajectory: Annotated[list[dict[str, Any]], _merge_trajectory]
    completed: bool
    pending_approval: dict[str, Any] | None  # зарезервовано для HITL (S09), поки завжди None
    thread_id: str  # зарезервовано для persistence (S07)
    agent_step_counts: dict[str, int]
    handoff_context: dict[str, Any] | None
    handoff_count: int


class RouteDecision(BaseModel):
    """Валідоване рішення supervisor про маршрутизацію запиту."""

    action: AgentName = Field(description="Куди маршрутизувати: billing, tech, researcher або general")
    reasoning: str = Field(description="Коротке пояснення рішення українською")


# --- Supervisor router: live (LLM) і demo (детермінований) -----------------

_SUPERVISOR_SYSTEM_PROMPT = """Ти — маршрутизатор туристичної підтримки. Визнач, який
спеціалізований агент має обробити запит користувача, і поверни структуроване рішення.

Агенти:
- billing: бюджет подорожі, вартість, бронювання готелю.
- tech: погода, маршрут, активності, операційні проблеми подорожі.
- researcher: візові правила, туристичні поради, музеї, безпека, транспортні правила (RAG).
- general: привітання, уточнення, запити поза доменом подорожей.

Приклади:
Запит: "Скільки коштуватиме подорож до Парижа на 3 дні для двох?" -> billing
Запит: "У Львові буде дощ, що вдягнути?" -> tech
Запит: "Які правила відвідування Лувру?" -> researcher
Запит: "Привіт! Що ти вмієш?" -> general
Запит: "Порахуй бюджет поїздки" -> billing

Завжди повертай action рівно одним із чотирьох значень і коротке reasoning українською.
"""


class DemoSupervisorRouter:
    """Детермінований маршрутизатор для тестів і демонстрацій без LLM.

    Проходить той самий граф, що й live-router — відрізняється лише
    джерелом `RouteDecision` (ключові слова замість LLM).
    """

    def decide(self, user_goal: str) -> RouteDecision:
        lower = user_goal.lower()
        # tech перевіряється ПЕРЕД billing: змішаний запит ("яка погода і скільки
        # це коштуватиме") має спершу потрапити до tech, щоб той сам вирішив і
        # ініціював реальний handoff до billing — інакше жоден запит ніколи не
        # проходив би через сценарій handoff (billing "перехоплював" би одразу).
        if any(k in lower for k in ("погод", "дощ", "актив", "маршрут", "одяг")):
            return RouteDecision(action="tech", reasoning="Запит стосується погоди, маршруту чи активностей.")
        if any(k in lower for k in ("бюджет", "вартість", "коштувати", "заброн", "готел")):
            return RouteDecision(action="billing", reasoning="Запит стосується бюджету або бронювання.")
        if any(k in lower for k in ("прав", "віз", "лувр", "музе", "безпек", "транспорт", "страхув", "крадіж")):
            return RouteDecision(
                action="researcher", reasoning="Запит стосується туристичних правил чи довідкової інформації."
            )
        return RouteDecision(action="general", reasoning="Запит не належить до жодної спеціалізованої категорії.")


class LiveSupervisorRouter:
    """Live-маршрутизатор: `with_structured_output(RouteDecision)` з фолбеком."""

    def __init__(self, llm: Any, timeout_seconds: float = 20.0):
        self.model = llm.with_structured_output(RouteDecision)
        self.timeout_seconds = timeout_seconds

    def decide(self, user_goal: str) -> RouteDecision:
        prompt = f"{_SUPERVISOR_SYSTEM_PROMPT}\n\nЗапит: {user_goal}"
        try:
            decision = call_with_timeout(self.model.invoke, self.timeout_seconds, prompt)
        except Exception as exc:  # обмежений фолбек: невалідний/зависший structured output
            return RouteDecision(
                action="general", reasoning=f"Фолбек: не вдалося отримати маршрутизацію ({exc})."
            )
        if decision.action not in ("billing", "tech", "researcher", "general"):
            return RouteDecision(action="general", reasoning="Фолбек: невалідне значення action.")
        return decision


# --- billing: planner -> executor -> replanner (з ДЗ2 plan_execute.py) -----


class BillingPlan(BaseModel):
    """Структурований план бюджетно-довідкових кроків (без book_hotel — S09)."""

    goal: str = Field(description="Головна ціль користувача щодо бюджету/бронювання")
    steps: list[str] = Field(
        description="Послідовність з 1-4 кроків: лише calculate_trip_cost і search_knowledge"
    )


class BillingReplanDecision(BaseModel):
    action: Literal["continue", "replan", "finish"] = Field(
        description="continue — наступний крок, replan — замінити залишок, finish — завершити"
    )
    updated_steps: list[str] | None = Field(
        default=None, description="Нові невиконані кроки лише для action=replan"
    )
    reasoning: str = Field(description="Стисле пояснення рішення українською")


def _encode_step(tool_name: str, **arguments: Any) -> str:
    return json.dumps({"tool": tool_name, "args": arguments}, ensure_ascii=False, sort_keys=True)


def _decode_step(step: str) -> tuple[str, dict[str, Any]]:
    payload = json.loads(step)
    return payload["tool"], payload["args"]


_BUDGET_ALERT_THRESHOLD_EUR = 500.0
_COST_PATTERN = re.compile(r":\s*([\d.]+)\s*EUR")


_BOOKING_ACTION_KEYWORDS = ("забронюй", "забронювати", "оформи бронювання", "оформити бронювання")


class DemoBillingPlanner:
    def invoke(self, goal: str) -> BillingPlan:
        lower = goal.lower()
        if any(k in lower for k in _BOOKING_ACTION_KEYWORDS):
            # Явний намір забронювати -> ЗМІШАНИЙ план: спершу безпечний
            # calculate_trip_cost (виконується одразу, без підтвердження),
            # потім book_hotel (RISKY_TOOLS — виконавець зупиниться саме на
            # цьому кроці для HITL, S09; approval_gate вирішить його долю).
            # Синтетичні демо-параметри бронювання ("Усі бронювання
            # синтетичні" — вимога плану).
            city = _extract_city(goal, default="Париж")
            days = _extract_days(goal, default=2)
            steps = [
                _encode_step("calculate_trip_cost", city=city, days=max(days, 1), travelers=1),
                _encode_step(
                    "book_hotel",
                    hotel_name="Hotel Lumiere",
                    check_in="2026-09-15",
                    nights=max(days, 1),
                    total_cost=280.0,
                    currency="EUR",
                ),
            ]
            return BillingPlan(goal=goal, steps=steps)

        city = _extract_city(goal, default="Париж")
        days = _extract_days(goal, default=3)
        travelers = _extract_travelers(goal)
        steps = [_encode_step("calculate_trip_cost", city=city, days=days, travelers=travelers)]
        if any(k in lower for k in ("скасув", "поліс", "бронюв")):
            steps.append(_encode_step("search_knowledge", query="політика скасування бронювання готелю"))
        return BillingPlan(goal=goal, steps=steps)


class DemoBillingReplanner:
    """Реагує на фактичний результат: зависокий бюджет -> додає крок з порадами."""

    def invoke(self, context: dict[str, Any]) -> BillingReplanDecision:
        current = context["current_step"]
        plan = context["plan"]
        results = context["results"]
        # Перевіряємо тригер replan ПЕРШИМ, навіть якщо плановий "хвіст" уже
        # вичерпано (current >= len(plan)) — replan тут означає ДОДАТИ новий
        # крок за фактичним результатом, а не лише замінити залишок; якщо цю
        # перевірку поставити після "current >= len(plan): finish", такий
        # додатковий крок ніколи не додасться для плану з одним кроком.
        latest = results[-1] if results else ""
        match = _COST_PATTERN.search(latest)
        already_has_tip_step = any("поради для економії" in step for step in plan[current:])
        if match and float(match.group(1)) > _BUDGET_ALERT_THRESHOLD_EUR and not already_has_tip_step:
            return BillingReplanDecision(
                action="replan",
                updated_steps=[_encode_step("search_knowledge", query="поради для економії бюджету подорожі")],
                reasoning=(
                    f"Орієнтовний бюджет ({match.group(1)} EUR) перевищує поріг "
                    f"{_BUDGET_ALERT_THRESHOLD_EUR:g} EUR — додано крок з порадами щодо економії."
                ),
            )
        if current >= len(plan):
            return BillingReplanDecision(action="finish", reasoning="Усі заплановані кроки виконано.")
        return BillingReplanDecision(action="continue", reasoning="План залишається актуальним.")


class BillingState(TypedDict, total=False):
    goal: str
    plan: list[str]
    current_step: int
    results: list[str]
    completed: bool
    pending_risky_step: dict[str, Any] | None  # S09: сигнал назовні, tool ще НЕ виконано


class BillingAgent:
    """Підграф billing: окремі planner/executor/replanner-вузли (S05)."""

    def __init__(
        self,
        provider: Literal["demo", "gemini"],
        llm: Any,
        logger: TrajectoryLogger,
        guard: ExecutionGuard,
        dispatch: Any,
    ):
        self.provider = provider
        self.logger = logger
        self.guard = guard
        self.dispatch = dispatch
        if provider == "gemini":
            self.planner_model = llm.with_structured_output(BillingPlan)
            self.replanner_model = llm.with_structured_output(BillingReplanDecision)
        else:
            self.planner_model = DemoBillingPlanner()
            self.replanner_model = DemoBillingReplanner()
        self.app = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(BillingState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("replanner", self._replanner_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "replanner")
        graph.add_conditional_edges(
            "replanner",
            lambda state: END if state.get("completed", False) else "executor",
            {"executor": "executor", END: END},
        )
        return graph.compile()

    def _planner_node(self, state: BillingState) -> dict[str, Any]:
        self.guard.check_limits(0)
        started = time.monotonic()
        if self.provider == "gemini":
            prompt = (
                "Створи план з 1-4 кроків, використовуючи лише calculate_trip_cost і "
                "search_knowledge (get_weather і book_hotel НЕ входять у роль billing).\n"
                f"USER_QUERY: {state['goal']}"
            )
            plan = call_with_timeout(self.planner_model.invoke, self.guard.timeout_seconds, prompt)
        else:
            plan = self.planner_model.invoke(state["goal"])
        duration_ms = int((time.monotonic() - started) * 1000)
        self.logger.log(
            "billing_planner", state["goal"], plan.model_dump(), duration_ms=duration_ms, agent_name="billing"
        )
        return {"plan": plan.steps, "current_step": 0, "results": [], "completed": False, "pending_risky_step": None}

    async def _executor_node(self, state: BillingState) -> dict[str, Any]:
        index = state["current_step"]
        if index >= len(state["plan"]):
            return {"completed": True}
        self.guard.check_limits(index)
        name, arguments = _decode_step(state["plan"][index])

        if name in RISKY_TOOLS:
            # Ризикова дія (S09): НЕ виконуємо тут жодного dispatch. Billing
            # завершує свій підграф і сигналізує pending_risky_step назовні
            # — MASApp._billing_node перетворює це на pending_approval, а
            # approval_gate ставить граф на паузу через interrupt() ДО
            # будь-якого виклику tool.
            self.logger.log(
                "billing_executor_pending_approval",
                state["plan"][index],
                f"Очікує підтвердження людини: {name}",
                tool_calls=[{"name": name, "args": arguments}],
                agent_name="billing",
            )
            return {"completed": True, "pending_risky_step": {"name": name, "arguments": arguments}}

        self.guard.register_call(name, arguments)
        started = time.monotonic()
        try:
            tool_result = await self.dispatch(name, arguments, "billing")
            result = f"{name}: {tool_result}"
        except ToolDispatchError as exc:
            result = str(exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        number = index + 1
        self.logger.log(
            "billing_executor",
            state["plan"][index],
            result,
            tool_calls=[{"name": name, "args": arguments}],
            duration_ms=duration_ms,
            agent_name="billing",
        )
        return {"current_step": number, "results": [*state["results"], f"Крок {number}: {result}"]}

    def _replanner_node(self, state: BillingState) -> dict[str, Any]:
        # Виконавець уже позначив completed=True (звичайне завершення плану
        # АБО очікування підтвердження ризикової дії, S09) — не запускаємо
        # ще один зайвий (демо/LLM) виклик replanner, який міг би про це не
        # знати й повернути "continue".
        if state.get("completed"):
            return {}
        context = {
            "goal": state["goal"],
            "plan": state["plan"],
            "current_step": state["current_step"],
            "results": state["results"],
        }
        started = time.monotonic()
        if self.provider == "gemini":
            prompt = (
                "Оціни прогрес виконання плану і поверни continue, replan або finish. "
                "Для replan updated_steps містить лише нові невиконані кроки.\n"
                + json.dumps(context, ensure_ascii=False)
            )
            decision = call_with_timeout(self.replanner_model.invoke, self.guard.timeout_seconds, prompt)
        else:
            decision = self.replanner_model.invoke(context)
        duration_ms = int((time.monotonic() - started) * 1000)
        self.logger.log(
            "billing_replanner", context, decision.model_dump(), duration_ms=duration_ms, agent_name="billing"
        )
        if decision.action == "finish":
            return {"completed": True}
        if decision.action == "replan" and decision.updated_steps:
            updated = state["plan"][: state["current_step"]] + decision.updated_steps
            return {"plan": updated}
        return {}

    async def run(self, goal: str) -> tuple[list[str], list[str], dict[str, Any] | None]:
        self.guard.reset()
        final = await self.app.ainvoke(
            {
                "goal": goal,
                "plan": [],
                "current_step": 0,
                "results": [],
                "completed": False,
                "pending_risky_step": None,
            }
        )
        return final["plan"], final["results"], final.get("pending_risky_step")


# --- tech: ReAct LLM -> tools -> LLM (з ДЗ1 agent.py) -----------------------

_TECH_SYSTEM_PROMPT = """Ти — оперативний туристичний агент підтримки маршруту. Використовуй
tools get_weather і find_activities для фактів, не вигадуй дані. Відповідай українською."""


class DemoTechDecisionModel:
    """Детермінована заміна LLM для tech (адаптовано з ДЗ1 DemoDecisionModel)."""

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        query = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
        called = [m.name for m in messages if isinstance(m, ToolMessage)]
        lower = str(query).lower()
        city = _extract_city(str(query), default="Львів")
        days = _extract_days(str(query), default=1)

        # Спеціальний сценарій перевіряє контрольовану зупинку повторюваних викликів.
        if "повторюй" in lower:
            return self._call("get_weather", {"city": city, "days": days})

        plan: list[tuple[str, dict[str, Any]]] = []
        if any(w in lower for w in ("погод", "дощ", "одяг", "план", "маршрут")):
            plan.append(("get_weather", {"city": city, "days": min(days, 7)}))
        if any(w in lower for w in ("актив", "культур", "чим зайнятися")) or "план" in lower:
            plan.append(("find_activities", {"city": city, "interest": "культура"}))
        for name, arguments in plan:
            if name not in called:
                return self._call(name, arguments)
        observations = [_message_text(m) for m in messages if isinstance(m, ToolMessage)]
        return AIMessage(content=" ".join(observations) or "Уточніть місто та мету запиту.")

    @staticmethod
    def _call(name: str, arguments: dict[str, Any]) -> AIMessage:
        return AIMessage(
            content="", tool_calls=[{"name": name, "args": arguments, "id": str(uuid.uuid4()), "type": "tool_call"}]
        )


class TechState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


class TechAgent:
    """Підграф tech: agent -> tools -> agent (перенесено з ДЗ1 TravelAgent)."""

    def __init__(
        self,
        provider: Literal["demo", "gemini"],
        llm: Any,
        logger: TrajectoryLogger,
        guard: ExecutionGuard,
        dispatch: Any,
    ):
        self.provider = provider
        self.logger = logger
        self.guard = guard
        self.dispatch = dispatch
        # self.tools — лише для JSON-схем bind_tools (щоб LLM знав аргументи
        # get_weather/find_activities); фактичне ВИКОНАННЯ завжди йде через
        # self.dispatch, який сам вирішує — локальний wrapper чи MCP-tool.
        self.tools = [TOOLS_BY_NAME["get_weather"], TOOLS_BY_NAME["find_activities"]]
        if provider == "gemini":
            self.decision_model = llm.bind_tools(self.tools)
        else:
            self.decision_model = DemoTechDecisionModel()
        self.app = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(TechState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", self._router)
        graph.add_edge("tools", "agent")
        return graph.compile()

    def _agent_node(self, state: TechState) -> dict[str, Any]:
        step_count = state.get("step_count", 0)
        self.guard.check_limits(step_count)
        started = time.monotonic()
        if self.provider == "gemini":
            prompt_messages = [HumanMessage(content=_TECH_SYSTEM_PROMPT)] + state["messages"]
            response = call_with_timeout(self.decision_model.invoke, self.guard.timeout_seconds, prompt_messages)
        else:
            response = self.decision_model.invoke(state["messages"])
        duration_ms = int((time.monotonic() - started) * 1000)
        self.guard.check_limits(step_count)  # повторна перевірка фіксує перевищення timeout під час виклику
        calls = getattr(response, "tool_calls", []) or []
        self.logger.log(
            "tech_agent",
            _message_text(state["messages"][-1]),
            _message_text(response),
            calls,
            duration_ms,
            agent_name="tech",
        )
        for call in calls:
            self.guard.register_call(call["name"], call["args"])
        return {"messages": [response], "step_count": step_count + 1}

    async def _tools_node(self, state: TechState) -> dict[str, Any]:
        step_count = state.get("step_count", 0)
        self.guard.check_limits(step_count)
        calls = state["messages"][-1].tool_calls
        started = time.monotonic()
        # register_call для детектора циклів уже виконано в _agent_node одразу
        # після генерації tool_calls — тут його НЕ повторюємо (інакше кожен
        # виклик рахувався б двічі і max_repeats спрацював би вдвічі швидше).
        outputs: list[ToolMessage] = []
        for call in calls:
            try:
                result = await self.dispatch(call["name"], call["args"], "tech")
                outputs.append(ToolMessage(content=result, name=call["name"], tool_call_id=call["id"]))
            except ToolDispatchError as exc:
                outputs.append(
                    ToolMessage(content=str(exc), name=call["name"], tool_call_id=call["id"], status="error")
                )
        duration_ms = int((time.monotonic() - started) * 1000)
        self.guard.check_limits(step_count)
        self.logger.log(
            "tech_tools",
            calls,
            [_message_text(m) for m in outputs],
            duration_ms=duration_ms,
            agent_name="tech",
        )
        return {"messages": outputs, "step_count": step_count + 1}

    @staticmethod
    def _router(state: TechState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    async def run(self, goal: str) -> tuple[str, list[str]]:
        self.guard.reset()
        final = await self.app.ainvoke({"messages": [HumanMessage(content=goal)], "step_count": 0})
        answer = _message_text(final["messages"][-1])
        used_tools = [m.name for m in final["messages"] if isinstance(m, ToolMessage)]
        return answer, used_tools


# --- researcher: LLM вирішує, чи потрібен search_knowledge ------------------

_RESEARCHER_SYSTEM_PROMPT = """Ти — туристичний дослідник правил і довідкової інформації.
Викликай tool search_knowledge лише тоді, коли запит справді стосується візових правил,
музеїв, транспорту, безпеки чи інших довідкових тем. Для арифметичних/бюджетних питань
tool НЕ викликай — просто поясни, що це поза твоєю роллю."""

_RESEARCHER_ARITHMETIC_KEYWORDS = ("бюджет", "вартість", "коштувати", "ціна")
_RESEARCHER_RAG_KEYWORDS = ("прав", "віз", "лувр", "музе", "безпек", "транспорт", "страхув", "крадіж")


class DemoResearcherDecisionModel:
    """Повертає (потрібен_rag, запит_для_search_knowledge)."""

    def invoke(self, query: str) -> tuple[bool, str]:
        lower = query.lower()
        looks_arithmetic = any(k in lower for k in _RESEARCHER_ARITHMETIC_KEYWORDS)
        looks_rag = any(k in lower for k in _RESEARCHER_RAG_KEYWORDS)
        needs_rag = looks_rag or not looks_arithmetic
        return needs_rag, query


class ResearcherAgent:
    def __init__(
        self,
        provider: Literal["demo", "gemini"],
        llm: Any,
        logger: TrajectoryLogger,
        guard: ExecutionGuard,
        dispatch: Any,
    ):
        self.provider = provider
        self.logger = logger
        self.guard = guard
        self.dispatch = dispatch
        # general_faq заповнюється MASApp.connect_mcp() (S06) з resource
        # faq://general; поки MCP не підключено — лишається None і fast-path
        # нижче просто не спрацьовує (RAG-шлях без змін від S05).
        self.general_faq: list[dict[str, str]] | None = None
        self.search_tool = TOOLS_BY_NAME["search_knowledge"]  # лише схема для bind_tools
        if provider == "gemini":
            self.decision_model = llm.bind_tools([self.search_tool])
        else:
            self.decision_model = DemoResearcherDecisionModel()

    def _match_faq(self, goal: str) -> dict[str, str] | None:
        """Дешевий лексичний пошук по завантаженому faq://general — перш ніж
        звертатися до RAG tool. Простий збіг значущого слова (>3 символів)
        між запитом і питанням з FAQ.

        Самопосилальний запис FAQ ("Де шукати детальні правила...", чия
        відповідь сама відсилає до search_knowledge) свідомо виключено з
        матчингу: інакше загальне слово на кшталт "правила" перехоплювало б
        запити, що насправді потребують RAG (напр. про Лувр) — знайдено й
        виправлено під час ручної перевірки MCP-демонстрації.
        """
        if not self.general_faq:
            return None
        lower = goal.lower()
        for entry in self.general_faq:
            if "search_knowledge" in entry["answer"]:
                continue
            question_words = {w for w in entry["question"].lower().split() if len(w) > 3}
            if any(word in lower for word in question_words):
                return entry
        return None

    async def run(self, goal: str) -> str:
        self.guard.reset()
        self.guard.check_limits(0)
        started = time.monotonic()

        faq_match = self._match_faq(goal)
        if faq_match:
            duration_ms = int((time.monotonic() - started) * 1000)
            answer = f"{faq_match['answer']} (джерело: faq://general — «{faq_match['question']}»)"
            self.logger.log("researcher_faq_resource", goal, answer, duration_ms=duration_ms, agent_name="researcher")
            return answer

        if self.provider == "gemini":
            prompt = f"{_RESEARCHER_SYSTEM_PROMPT}\n\nЗапит: {goal}"
            response = call_with_timeout(self.decision_model.invoke, self.guard.timeout_seconds, prompt)
            duration_ms = int((time.monotonic() - started) * 1000)
            self.guard.check_limits(0)
            calls = getattr(response, "tool_calls", []) or []
            if not calls:
                answer = _message_text(response) or "Це питання не потребує пошуку в базі знань."
                self.logger.log("researcher_decision", goal, answer, duration_ms=duration_ms, agent_name="researcher")
                return answer
            call = calls[0]
            self.guard.register_call(call["name"], call["args"])
            try:
                result = await self.dispatch(call["name"], call["args"], "researcher")
            except ToolDispatchError as exc:
                self.logger.log("researcher_rag_error", goal, str(exc), duration_ms=duration_ms, agent_name="researcher")
                return str(exc)
            self.logger.log(
                "researcher_rag", goal, result, tool_calls=calls, duration_ms=duration_ms, agent_name="researcher"
            )
            return f"На основі довідкових матеріалів:\n{result}"

        needs_rag, query = self.decision_model.invoke(goal)
        duration_ms = int((time.monotonic() - started) * 1000)
        self.guard.check_limits(0)
        if not needs_rag:
            answer = "Це питання не стосується довідкових правил — RAG тут не потрібен; зверніться до billing/tech."
            self.logger.log("researcher_decision", goal, answer, duration_ms=duration_ms, agent_name="researcher")
            return answer
        self.guard.register_call("search_knowledge", {"query": query})
        try:
            result = await self.dispatch("search_knowledge", {"query": query}, "researcher")
        except ToolDispatchError as exc:
            self.logger.log("researcher_rag_error", goal, str(exc), duration_ms=duration_ms, agent_name="researcher")
            return str(exc)
        self.logger.log(
            "researcher_rag",
            goal,
            result,
            tool_calls=[{"name": "search_knowledge", "args": {"query": query}}],
            duration_ms=duration_ms,
            agent_name="researcher",
        )
        return f"На основі довідкових матеріалів:\n{result}"


# --- general: без ризикових викликів -----------------------------------------


class GeneralAgent:
    def __init__(self, provider: Literal["demo", "gemini"], llm: Any, logger: TrajectoryLogger):
        self.provider = provider
        self.llm = llm
        self.logger = logger
        # mcp_client заповнюється MASApp.connect_mcp() (S06); поки None —
        # поведінка ідентична S05 (canned demo-рядок чи прямий LLM-виклик).
        self.mcp_client: Any = None

    async def run(self, goal: str) -> str:
        started = time.monotonic()
        if self.mcp_client is not None:
            # prompt повертає ШАБЛОН (не готову вигадану відповідь) —
            # природний фіт для ролі general: "привітання й уточнення".
            messages = await self.mcp_client.get_prompt(
                "travel",
                "travel_reply",
                arguments={"traveler_name": "Мандрівник", "issue_summary": goal, "tone": "дружній"},
            )
            answer = str(messages[0].content) if messages else ""
            duration_ms = int((time.monotonic() - started) * 1000)
            self.logger.log("general_prompt_template", goal, answer, duration_ms=duration_ms, agent_name="general")
            return answer
        if self.provider == "gemini":
            prompt = (
                "Ти — привітний туристичний асистент. Дай коротку відповідь або постав "
                "уточнююче питання. НЕ використовуй жодних tools.\n" + goal
            )
            response = call_with_timeout(self.llm.invoke, 15.0, prompt)
            answer = _message_text(response)
        else:
            answer = (
                "Привіт! Я можу допомогти з бюджетом подорожі, погодою й активностями "
                "або туристичними правилами. Уточніть, будь ласка, ваш запит."
            )
        duration_ms = int((time.monotonic() - started) * 1000)
        self.logger.log("general", goal, answer, duration_ms=duration_ms, agent_name="general")
        return answer


# --- MASApp: supervisor + чотири агенти в одному LangGraph -----------------


class MASApp:
    def __init__(
        self,
        provider: Literal["demo", "gemini"] = "demo",
        run_id: str = "mas-run",
        checkpointer: Any = None,
        interrupt_after: list[str] | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.provider = provider
        self.logger = TrajectoryLogger(run_id)
        self.llm: Any = None
        self.mcp_client: Any = None
        # checkpointer/interrupt_after — durable state (S07). Без checkpointer
        # (типовий випадок для S01–S06 і більшості тестів) граф поводиться як
        # раніше: без збереження стану між викликами. З файловим SqliteSaver
        # (persistence_demo.py) той самий граф переживає перезапуск процесу.
        self.checkpointer = checkpointer
        self.interrupt_after = interrupt_after
        # ОДИН екземпляр на весь MASApp (S08): rolling window не обнуляється
        # на кожен вузол графа — перевіряється рівно один раз на вхід запиту,
        # у handle_query. Тести підставляють rate_limiter із фейковим clock.
        self.rate_limiter = rate_limiter or RateLimiter()
        # Локальні LangChain-обгортки за замовчуванням; connect_mcp() (S06)
        # ОНОВЛЮЄ цей самий словник in-place MCP-tools з тими самими іменами
        # — агенти тримають лише bound-метод dispatch_tool, тому підміна
        # діє одразу, без перестворення агентів.
        self.tools_by_name: dict[str, Any] = dict(TOOLS_BY_NAME)

        if provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("Для режиму gemini задайте змінну GOOGLE_API_KEY")
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            self.router: Any = LiveSupervisorRouter(self.llm)
        else:
            self.router = DemoSupervisorRouter()

        self.guards: dict[str, ExecutionGuard] = {
            "billing": ExecutionGuard(max_steps=6, timeout_seconds=30.0, max_repeats=3),
            "tech": ExecutionGuard(max_steps=6, timeout_seconds=30.0, max_repeats=3),
            "researcher": ExecutionGuard(max_steps=4, timeout_seconds=20.0, max_repeats=3),
        }
        self.billing_agent = BillingAgent(provider, self.llm, self.logger, self.guards["billing"], self.dispatch_tool)
        self.tech_agent = TechAgent(provider, self.llm, self.logger, self.guards["tech"], self.dispatch_tool)
        self.researcher_agent = ResearcherAgent(
            provider, self.llm, self.logger, self.guards["researcher"], self.dispatch_tool
        )
        self.general_agent = GeneralAgent(provider, self.llm, self.logger)

        self.app = self._build_graph()

    # -- MCP (S06) -------------------------------------------------------------

    async def connect_mcp(self, server_path: str | Path | None = None, env: dict[str, str] | None = None) -> None:
        """Підключити travel-support MCP-сервер (S03) через stdio.

        Абсолютний шлях до `mcp_server.py` і поточний `sys.executable` (а не
        системний `python` з PATH) — інакше сервер міг би стартувати в
        іншому середовищі без потрібних залежностей. Після виклику tools з
        MCP ЗАМІНЮЮТЬ локальні wrappers у `self.tools_by_name` для тих самих
        імен — виконання відбувається реальним subprocess-викликом MCP
        (`MultiServerMCPClient` відкриває нову сесію на кожен виклик tool,
        resource чи prompt — довготривалого з'єднання тримати не потрібно).
        Також завантажує resource `faq://general` у researcher.

        `env`, якщо переданий, ПОВНІСТЮ замінює середовище subprocess (а не
        додає до нього) — виклик має сам змержити його з `os.environ`,
        якщо потрібно зберегти PATH/локаль тощо (див. `__main__` нижче).
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient

        path = Path(server_path) if server_path else Path(__file__).resolve().parent / "mcp_server.py"
        connection: dict[str, Any] = {"command": sys.executable, "args": [str(path)], "transport": "stdio"}
        if env is not None:
            connection["env"] = env
        # handle_tool_errors=False: за замовчуванням адаптер ковтає
        # CallToolResult(isError=True) і повертає текст помилки як звичайний
        # успішний рядок (щоб ReAct-агент міг "самовиправитись") — саме
        # той фальшивий успіх, якого явно вимагає уникати план. З False
        # такий збій підіймається як виключення й ловиться dispatch_tool
        # так само, як помилка локального tool (виявлено emпірично і
        # виправлено — інакше невалідне місто виглядало б як "успішний"
        # прогноз погоди).
        self.mcp_client = MultiServerMCPClient({"travel": connection}, handle_tool_errors=False)
        mcp_tools = await self.mcp_client.get_tools()
        self.tools_by_name.update({tool.name: tool for tool in mcp_tools})

        faq_blobs = await self.mcp_client.get_resources("travel", uris="faq://general")
        general_faq = json.loads(faq_blobs[0].as_string()) if faq_blobs else None
        self.researcher_agent.general_faq = general_faq
        self.general_agent.mcp_client = self.mcp_client

        self.logger.log(
            "mcp_connect",
            str(path),
            f"Завантажено {len(mcp_tools)} MCP tools ({sorted(t.name for t in mcp_tools)}); "
            f"faq://general: {len(general_faq) if general_faq else 0} записів.",
            agent_name="supervisor",
        )

    async def dispatch_tool(self, name: str, arguments: dict[str, Any], agent_name: str) -> str:
        """ЄДИНА точка виклику будь-якого tool з будь-якого агента (billing/
        tech/researcher) — локального чи MCP-tool з `self.tools_by_name`.

        Guardrail-політики (S08), у порядку застосування:
        1. `tool_guardrail(agent_name, name)` — allowlist per agent,
           перевіряється ПЕРЕД будь-яким local/MCP викликом. `agent_name`
           надходить з контрольованого execution context (кожен агент
           передає власне ім'я явно), а не з фрази користувача.
        2. Виконання tool (local wrapper чи MCP).
        3. `input_guardrail` на результат — tool/MCP resource/RAG повертають
           недовірений зовнішній текст, який далі потрапляє у контекст LLM
           (напр. billing/tech передають попередні `results` у наступний
           LLM-виклик); підозрілий вміст блокується так само, як вхід
           користувача.

        Записує tool_call_id, аргументи, тривалість і agent_name у спільний
        TrajectoryLogger — і для успіху, і для помилки. Збій (заборонено
        guardrail, невідомий tool, падіння MCP-сервера, підозрілий вміст
        результату) підіймається як `ToolDispatchError`, а НЕ повертається
        як фальшивий успішний рядок.
        """
        tool_call_id = str(uuid.uuid4())
        started = time.monotonic()

        def _deny(message: str) -> None:
            duration_ms = int((time.monotonic() - started) * 1000)
            self.logger.log(
                "tool_dispatch_error",
                {"tool_call_id": tool_call_id, "name": name, "args": arguments},
                message,
                tool_calls=[{"name": name, "args": arguments, "id": tool_call_id}],
                duration_ms=duration_ms,
                agent_name=agent_name,
            )
            raise ToolDispatchError(message)

        if not tool_guardrail(agent_name, name):
            _deny(f"Guardrail: агенту '{agent_name}' заборонено викликати tool '{name}'.")

        tool = self.tools_by_name.get(name)
        if tool is None:
            _deny(f"Невідомий інструмент: {name}.")

        try:
            if hasattr(tool, "ainvoke"):
                raw_result = await tool.ainvoke(arguments)
            else:
                raw_result = tool.invoke(arguments)
            result = _stringify_tool_result(raw_result)
        except Exception as exc:  # збій transport/сервера чи бізнес-логіки — не ковтаємо мовчки
            _deny(f"Tool {name} завершився помилкою: {exc}")

        is_safe, sanitized_or_error = input_guardrail(result, max_len=20_000)
        if not is_safe:
            _deny(f"Guardrail: результат tool '{name}' заблоковано — {sanitized_or_error}")
        result = sanitized_or_error

        duration_ms = int((time.monotonic() - started) * 1000)
        self.logger.log(
            "tool_dispatch",
            {"tool_call_id": tool_call_id, "name": name, "args": arguments},
            result,
            tool_calls=[{"name": name, "args": arguments, "id": tool_call_id}],
            duration_ms=duration_ms,
            agent_name=agent_name,
        )
        return result

    # -- побудова зовнішнього графа ------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(MASState)
        graph.add_node("supervisor", self._node("supervisor", self._supervisor_node))
        graph.add_node("billing", self._node("billing", self._billing_node))
        graph.add_node("tech", self._node("tech", self._tech_node))
        graph.add_node("researcher", self._node("researcher", self._researcher_node))
        graph.add_node("general", self._node("general", self._general_node))
        graph.add_node("approval_gate", self._node("approval_gate", self._approval_gate_node))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "billing": "billing",
                "tech": "tech",
                "researcher": "researcher",
                "general": "general",
                "approval_gate": "approval_gate",
                END: END,
            },
        )
        graph.add_edge("billing", "supervisor")
        graph.add_edge("tech", "supervisor")
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("general", "supervisor")
        graph.add_edge("approval_gate", "supervisor")
        return graph.compile(checkpointer=self.checkpointer, interrupt_after=self.interrupt_after)

    def _node(self, _name: str, fn: Any) -> Any:
        """Обгортка: додає до `update["trajectory"]` лише нові кроки, залишені
        цим викликом у спільному self.logger — без дублювання (reducer вище
        додатково підстраховує ідемпотентність при повторних візитах). Також
        маскує PII у `results`/`messages` через `output_guardrail` (S08) —
        ОДНЕ місце для всіх agent-вузлів, а не окремий виклик у кожному.

        `fn` може бути sync (supervisor/general без MCP) або async (billing/
        tech/researcher — викликають dispatch_tool); LangGraph підтримує
        змішані sync/async вузли в одному графі при виклику через `ainvoke`,
        тому тип обгортки визначаємо автоматично.
        """
        if asyncio.iscoroutinefunction(fn):

            async def wrapped_async(state: MASState) -> dict[str, Any]:
                before = len(self.logger.steps)
                update = await fn(state)
                update = self._apply_output_guardrail(update)
                delta = self.logger.steps[before:]
                if delta:
                    update = {**update, "trajectory": delta}
                return update

            return wrapped_async

        def wrapped_sync(state: MASState) -> dict[str, Any]:
            before = len(self.logger.steps)
            update = fn(state)
            update = self._apply_output_guardrail(update)
            delta = self.logger.steps[before:]
            if delta:
                update = {**update, "trajectory": delta}
            return update

        return wrapped_sync

    @staticmethod
    def _apply_output_guardrail(update: dict[str, Any]) -> dict[str, Any]:
        """Маскує PII (email/телефон/картка/IBAN/ІПН) у `results`/`messages`
        ПЕРЕД поверненням у стан — S08, застосовується до кожного agent-вузла.
        """
        if update.get("results"):
            update = {**update, "results": [output_guardrail(item)[0] for item in update["results"]]}
        if update.get("messages"):
            redacted_messages = []
            for message in update["messages"]:
                redacted_content, _pii_found = output_guardrail(_message_text(message))
                redacted_messages.append(message.model_copy(update={"content": redacted_content}))
            update = {**update, "messages": redacted_messages}
        return update

    @staticmethod
    def _user_goal(state: MASState) -> str:
        # Ціль читаємо з HumanMessage, а НЕ з останнього ToolMessage — інакше
        # маршрутизація випадково залежала б від останнього спостереження tool.
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    @staticmethod
    def _route_from_supervisor(state: MASState) -> str:
        if state.get("completed"):
            return END
        return state["current_agent"]

    # -- supervisor ----------------------------------------------------------

    def _supervisor_node(self, state: MASState) -> dict[str, Any]:
        # Кожен агент-вузол безумовно веде назад до supervisor (щоб той міг
        # перевірити handoff), тому цей вузол виконується ще раз навіть після
        # completed=True. Без цього short-circuit він би перезаписав
        # current_agent новим (уже зайвим) рішенням маршрутизації.
        if state.get("completed"):
            return {}

        step_count = state.get("step_count", 0) + 1
        if step_count > MAX_SUPERVISOR_VISITS:
            message = f"Досягнуто глобальний ліміт кроків supervisor ({MAX_SUPERVISOR_VISITS})."
            self.logger.log("supervisor_limit", state.get("results", []), message, agent_name="supervisor")
            return {
                "current_agent": "general",
                "completed": True,
                "step_count": step_count,
                "results": [message],
                "messages": [AIMessage(content=message)],
            }

        # Ризикова дія очікує підтвердження (S09) — маршрутизуємо ПРЯМО до
        # approval_gate, без повторної LLM/keyword-класифікації (той самий
        # принцип, що й для handoff нижче: явний контрольований стан
        # переважає над router.decide).
        if state.get("pending_approval"):
            return {"current_agent": "approval_gate", "step_count": step_count}

        handoff = state.get("handoff_context")
        handoff_count = state.get("handoff_count", 0)
        if handoff and handoff_count < MAX_HANDOFFS:
            self.logger.log(
                "supervisor_handoff", handoff, f"handoff -> {handoff['target']}", agent_name="supervisor"
            )
            return {"current_agent": handoff["target"], "step_count": step_count, "handoff_count": handoff_count + 1}

        goal = self._user_goal(state)
        decision = self.router.decide(goal)
        self.logger.log("supervisor_route", goal, decision.model_dump(), agent_name="supervisor")
        return {"current_agent": decision.action, "step_count": step_count}

    # -- агенти-вузли ----------------------------------------------------------

    async def _billing_node(self, state: MASState) -> dict[str, Any]:
        handoff = state.get("handoff_context")
        goal = handoff["context"] if handoff else self._user_goal(state)
        try:
            plan, results, pending_risky_step = await self.billing_agent.run(goal)
        except SafetyStop as exc:
            message = f"billing: безпечно зупинено ({exc})"
            return {
                "completed": True,
                "handoff_context": None,
                "results": [message],
                "messages": [AIMessage(content=message)],
            }

        if pending_risky_step is not None:
            # S09: НЕ completed — граф має пройти через approval_gate.
            tool_name = pending_risky_step["name"]
            args = pending_risky_step["arguments"]
            operation_id = book_hotel_operation_id(**args) if tool_name == "book_hotel" else None
            pending_approval = {
                "agent_name": "billing",
                "tool": tool_name,
                "args": args,
                "consequences": (
                    "Ризикова дія з фінансовими наслідками (демонстраційне бронювання, синтетичні дані)."
                ),
                "operation_id": operation_id,
            }
            return {
                "completed": False,
                "plan": plan,
                "results": results,
                "pending_approval": pending_approval,
                "messages": [AIMessage(content=f"Очікую підтвердження людини для {tool_name}...")],
                "handoff_context": None,
            }

        answer = "\n".join(results) if results else "Не вдалося опрацювати бюджетний запит."
        return {
            "completed": True,
            "plan": plan,
            "current_step": len(results),  # відображає прогрес billing у зовнішньому стані (S07: порівняння до/після)
            "results": results,
            "messages": [AIMessage(content=answer)],
            "handoff_context": None,
        }

    async def _approval_gate_node(self, state: MASState) -> dict[str, Any]:
        """HITL approval gate (S09): interrupt() ставить граф на паузу ДО
        будь-якого виклику ризикового tool. Рішення надходить через
        `Command(resume={"action": "approve"|"reject"|"edit", ...})` для
        того самого thread_id (потребує checkpointer — `MASApp(checkpointer=...)`).
        """
        pending = state.get("pending_approval")
        if not pending:
            return {"completed": True}

        decision = interrupt(
            {
                "message": "Підтвердіть ризикову дію перед виконанням.",
                "agent_name": pending["agent_name"],
                "tool": pending["tool"],
                "args": pending["args"],
                "consequences": pending["consequences"],
                "operation_id": pending["operation_id"],
            }
        )

        # Boolean approved=True (LLM міг би сам його підставити) НЕ є людським
        # підтвердженням — приймаємо лише явний dict з дозволеною action.
        if not is_valid_decision(decision):
            message = "Некоректне рішення підтвердження — бронювання контрольовано відхилено."
            self.logger.log("approval_gate_invalid_decision", pending, message, agent_name="billing")
            return {
                "completed": True,
                "pending_approval": None,
                "results": [message],
                "messages": [AIMessage(content=message)],
            }

        tool_name = pending["tool"]
        args = dict(pending["args"])
        agent_name = pending["agent_name"]
        action = decision["action"]

        if action == "reject":
            reason = decision.get("reason", "без пояснення")
            message = f"Бронювання відхилено оператором: {reason}. Tool не виконувався."
            self.logger.log("approval_gate_reject", pending, message, agent_name=agent_name)
            return {
                "completed": True,
                "pending_approval": None,
                "results": [message],
                "messages": [AIMessage(content=message)],
            }

        if action == "edit":
            edits = decision.get("args", {})
            args.update(edits)
            try:
                validated = BookingInput(**args)
            except ValidationError as exc:
                message = f"Редагування невалідне — {exc}. Бронювання не виконано."
                self.logger.log("approval_gate_invalid_edit", {**pending, "edits": edits}, message, agent_name=agent_name)
                return {
                    "completed": True,
                    "pending_approval": None,
                    "results": [message],
                    "messages": [AIMessage(content=message)],
                }
            args = validated.model_dump()

        # approve (з оригінальними чи відредагованими й перевалідованими args)
        try:
            tool_result = await self.dispatch_tool(tool_name, args, agent_name)
            message = f"{tool_name}: {tool_result}"
        except ToolDispatchError as exc:
            message = str(exc)
        self.logger.log(f"approval_gate_{action}", {**pending, "final_args": args}, message, agent_name=agent_name)
        return {
            "completed": True,
            "pending_approval": None,
            "results": [message],
            "messages": [AIMessage(content=message)],
        }

    async def _tech_node(self, state: MASState) -> dict[str, Any]:
        goal = self._user_goal(state)
        try:
            answer, _used_tools = await self.tech_agent.run(goal)
        except SafetyStop as exc:
            message = f"tech: безпечно зупинено ({exc})"
            return {"completed": True, "results": [message], "messages": [AIMessage(content=message)]}

        lower = goal.lower()
        needs_billing = any(k in lower for k in ("бюджет", "вартість", "коштувати"))
        if needs_billing and state.get("handoff_count", 0) < MAX_HANDOFFS:
            return {
                "completed": False,
                "results": [f"tech: {answer}"],
                "messages": [AIMessage(content=answer)],
                "handoff_context": {"target": "billing", "context": goal},
            }
        return {"completed": True, "results": [f"tech: {answer}"], "messages": [AIMessage(content=answer)]}

    async def _researcher_node(self, state: MASState) -> dict[str, Any]:
        goal = self._user_goal(state)
        try:
            answer = await self.researcher_agent.run(goal)
        except SafetyStop as exc:
            answer = f"researcher: безпечно зупинено ({exc})"
        return {"completed": True, "results": [f"researcher: {answer}"], "messages": [AIMessage(content=answer)]}

    async def _general_node(self, state: MASState) -> dict[str, Any]:
        goal = self._user_goal(state)
        answer = await self.general_agent.run(goal)
        return {"completed": True, "results": [f"general: {answer}"], "messages": [AIMessage(content=answer)]}

    # -- публічний вхід --------------------------------------------------------

    @staticmethod
    def initial_state(query: str, thread_id: str = "demo") -> MASState:
        return {
            "messages": [HumanMessage(content=query)],
            "current_agent": "general",
            "plan": [],
            "current_step": 0,
            "results": [],
            "step_count": 0,
            "trajectory": [],
            "completed": False,
            "pending_approval": None,
            "thread_id": thread_id,
            "agent_step_counts": {},
            "handoff_context": None,
            "handoff_count": 0,
        }

    @staticmethod
    def config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    async def handle_query(self, query: str, thread_id: str = "demo") -> dict[str, Any]:
        """Вхід у MAS (S08): rate-limit і input guardrail перевіряються ТУТ,
        ДО побудови стану й виклику графа — заблокований запит жодного
        вузла/tool не торкається (не лише повертає "безпечну" відповідь
        post-hoc). `thread_id` використовується і як `session_id` для
        rate-limit (одна сесія — один потік розмови).
        """
        allowed, rate_message = self.rate_limiter.check(thread_id)
        if not allowed:
            return self._blocked_state(query, thread_id, "rate_limit_guardrail", rate_message)

        is_safe, sanitized_or_error = input_guardrail(query)
        if not is_safe:
            return self._blocked_state(query, thread_id, "input_guardrail", sanitized_or_error)

        return await self.app.ainvoke(self.initial_state(sanitized_or_error, thread_id), config=self.config(thread_id))

    def _blocked_state(self, query: str, thread_id: str, guardrail_name: str, message: str) -> MASState:
        self.logger.log(guardrail_name, query, message, agent_name="supervisor")
        blocked_answer = f"Запит заблоковано ({guardrail_name}): {message}"
        state = self.initial_state(query, thread_id)
        state["completed"] = True
        state["results"] = [blocked_answer]
        state["messages"] = [*state["messages"], AIMessage(content=blocked_answer)]
        return state

    async def resume(self, thread_id: str) -> dict[str, Any]:
        """Продовжити граф із checkpoint, збереженого для `thread_id` (S07).

        `None` як вхід — стандартний спосіб LangGraph "продовжити з останнього
        checkpoint" (потребує підключеного `checkpointer`); працює однаково,
        викликаний з того самого процесу чи з нового (checkpointer читає стан
        із файлу, а не з пам'яті поточного об'єкта).
        """
        if self.checkpointer is None:
            raise ValueError("resume() потребує MASApp(checkpointer=...) — durable state не увімкнено.")
        return await self.app.ainvoke(None, config=self.config(thread_id))

    async def get_snapshot(self, thread_id: str) -> dict[str, Any]:
        """Знімок поточного checkpoint-стану для `thread_id` (без виконання
        графа) — для порівняння plan/current_step/results до і після
        паузи/відновлення. Async — `AsyncSqliteSaver` не підтримує sync
        `get_state()` (симетрично до того, що синхронний `SqliteSaver` не
        підтримує `ainvoke`)."""
        if self.checkpointer is None:
            raise ValueError("get_snapshot() потребує MASApp(checkpointer=...).")
        snapshot = await self.app.aget_state(self.config(thread_id))
        return dict(snapshot.values) if snapshot else {}


async def _run_demo() -> None:
    from tools_legacy import configure_storage

    # Демо-запуск — це runtime-сесія, не контрольний артефакт здачі: сховище
    # спрямовуємо у gitignored runtime/, а не в корінь репозиторію. Той самий
    # runtime_dir передається MCP-subprocess'у через env, щоб і його ChromaDB/
    # bookings.json теж лишались поза коренем репозиторію.
    runtime_dir = Path(__file__).resolve().parent / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    configure_storage(runtime_dir / "chroma_db", runtime_dir / "bookings.json")

    mas = MASApp(provider="demo", run_id="mas-demo")
    # provider="demo" лишає маршрутизацію й планування детермінованими
    # (без LLM/API-ключа), а connect_mcp() підміняє ВИКОНАННЯ tools на
    # справжні виклики subprocess MCP-сервера — і те, і інше проходить
    # через той самий граф і той самий dispatch_tool.
    mcp_env = {
        **os.environ,
        "MCP_KNOWLEDGE_PATH": str(runtime_dir / "chroma_db"),
        "MCP_BOOKING_PATH": str(runtime_dir / "bookings.json"),
    }
    await mas.connect_mcp(env=mcp_env)

    demo_queries = [
        "Скільки коштуватиме подорож до Парижа на 5 днів для двох?",  # billing + реальний replan (>500 EUR), MCP
        "У Львові дощ: перевір погоду і підбери культурні активності",  # tech, MCP
        "Які правила відвідування Лувру?",  # researcher: RAG через MCP (не FAQ resource)
        "Як зменшити ризик крадіжок у подорожі?",  # researcher: fast-path через faq://general resource
        "Привіт! Що ти вмієш?",  # general: prompt travel_reply через MCP
        "Яка погода у Львові і скільки коштуватиме подорож на 3 дні?",  # tech -> handoff -> billing, обидва MCP
        "повторюй погоду у Львові",  # tech: контрольована зупинка через детектор циклів
    ]
    for query in demo_queries:
        print(f"\n=== {query} ===")
        final_state = await mas.handle_query(query)
        print("current_agent:", final_state.get("current_agent"))
        print("results:", final_state.get("results"))

    from trajectory_logger import save_trajectories

    save_trajectories([mas.logger.as_dict()], runtime_dir / "trajectory_demo_s06.json")
    print(f"\nТраєкторію збережено у {runtime_dir / 'trajectory_demo_s06.json'} (demo-режим, tools через MCP).")


if __name__ == "__main__":
    asyncio.run(_run_demo())
