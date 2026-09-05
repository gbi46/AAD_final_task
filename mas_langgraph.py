"""Мультиагентна система (MAS) туристичної координації над LangGraph (S05).

Supervisor-вузол маршрутизує запит між чотирма спеціалізованими агентами:

- `billing` — бюджет і бронювання: підграф planner → executor → replanner,
  перенесений і адаптований з `Kravchuk_AAD_2/plan_execute.py` (без
  persistence/HITL — це S07/S09; `book_hotel` тут НЕ викликається
  автоматично, лише `calculate_trip_cost`/`search_knowledge`).
- `tech` — погода, маршрут, активності: ReAct LLM → tools → LLM,
  перенесений і адаптований з `Kravchuk_AAD_1/agent.py`.
- `researcher` — туристичні правила й FAQ: LLM вирішує, чи потрібен
  `search_knowledge` (Agentic RAG), відповідь цитує ID документів.
- `general` — привітання й уточнення без ризикових викликів.

Локальні валідовані wrappers з `tools_legacy.py` викликаються напряму;
підключення того самого набору tools через MCP — S06.
"""

from __future__ import annotations

import json
import operator
import os
import re
import time
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from safety import ExecutionGuard, SafetyStop, call_with_timeout
from tools_legacy import TOOLS_BY_NAME, search_knowledge_core
from trajectory_logger import TrajectoryLogger

AgentName = Literal["billing", "tech", "researcher", "general"]

MAX_HANDOFFS = 1
MAX_SUPERVISOR_VISITS = 8


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
        if any(k in lower for k in ("прав", "віз", "лувр", "музе", "безпек", "транспорт", "страхув")):
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


class DemoBillingPlanner:
    def invoke(self, goal: str) -> BillingPlan:
        city = _extract_city(goal, default="Париж")
        days = _extract_days(goal, default=3)
        travelers = _extract_travelers(goal)
        steps = [_encode_step("calculate_trip_cost", city=city, days=days, travelers=travelers)]
        if any(k in goal.lower() for k in ("скасув", "поліс", "бронюв")):
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


class BillingAgent:
    """Підграф billing: окремі planner/executor/replanner-вузли (S05)."""

    def __init__(self, provider: Literal["demo", "gemini"], llm: Any, logger: TrajectoryLogger, guard: ExecutionGuard):
        self.provider = provider
        self.logger = logger
        self.guard = guard
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
        return {"plan": plan.steps, "current_step": 0, "results": [], "completed": False}

    def _executor_node(self, state: BillingState) -> dict[str, Any]:
        index = state["current_step"]
        if index >= len(state["plan"]):
            return {"completed": True}
        self.guard.check_limits(index)
        name, arguments = _decode_step(state["plan"][index])
        started = time.monotonic()
        if name not in TOOLS_BY_NAME:
            result = f"Невідомий інструмент: {name}."
        else:
            self.guard.register_call(name, arguments)
            result = f"{name}: {TOOLS_BY_NAME[name].invoke(arguments)}"
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

    def run(self, goal: str) -> tuple[list[str], list[str]]:
        self.guard.reset()
        final = self.app.invoke(
            {"goal": goal, "plan": [], "current_step": 0, "results": [], "completed": False}
        )
        return final["plan"], final["results"]


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

    def __init__(self, provider: Literal["demo", "gemini"], llm: Any, logger: TrajectoryLogger, guard: ExecutionGuard):
        self.provider = provider
        self.logger = logger
        self.guard = guard
        self.tools = [TOOLS_BY_NAME["get_weather"], TOOLS_BY_NAME["find_activities"]]
        if provider == "gemini":
            self.decision_model = llm.bind_tools(self.tools)
        else:
            self.decision_model = DemoTechDecisionModel()
        self.tool_node = ToolNode(self.tools)
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

    def _tools_node(self, state: TechState) -> dict[str, Any]:
        step_count = state.get("step_count", 0)
        self.guard.check_limits(step_count)
        started = time.monotonic()
        result = self.tool_node.invoke({"messages": state["messages"]})
        duration_ms = int((time.monotonic() - started) * 1000)
        self.guard.check_limits(step_count)
        outputs = result["messages"]
        self.logger.log(
            "tech_tools",
            state["messages"][-1].tool_calls,
            [_message_text(m) for m in outputs],
            duration_ms=duration_ms,
            agent_name="tech",
        )
        return {"messages": outputs, "step_count": step_count + 1}

    @staticmethod
    def _router(state: TechState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    def run(self, goal: str) -> tuple[str, list[str]]:
        self.guard.reset()
        final = self.app.invoke({"messages": [HumanMessage(content=goal)], "step_count": 0})
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
    def __init__(self, provider: Literal["demo", "gemini"], llm: Any, logger: TrajectoryLogger, guard: ExecutionGuard):
        self.provider = provider
        self.logger = logger
        self.guard = guard
        self.search_tool = TOOLS_BY_NAME["search_knowledge"]
        if provider == "gemini":
            self.decision_model = llm.bind_tools([self.search_tool])
        else:
            self.decision_model = DemoResearcherDecisionModel()

    def run(self, goal: str) -> str:
        self.guard.reset()
        self.guard.check_limits(0)
        started = time.monotonic()
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
            result = search_knowledge_core(**call["args"])
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
        result = search_knowledge_core(query, limit=3)
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

    def run(self, goal: str) -> str:
        started = time.monotonic()
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
    def __init__(self, provider: Literal["demo", "gemini"] = "demo", run_id: str = "mas-run"):
        self.provider = provider
        self.logger = TrajectoryLogger(run_id)
        self.llm: Any = None

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
        self.billing_agent = BillingAgent(provider, self.llm, self.logger, self.guards["billing"])
        self.tech_agent = TechAgent(provider, self.llm, self.logger, self.guards["tech"])
        self.researcher_agent = ResearcherAgent(provider, self.llm, self.logger, self.guards["researcher"])
        self.general_agent = GeneralAgent(provider, self.llm, self.logger)

        self.app = self._build_graph()

    # -- побудова зовнішнього графа ------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(MASState)
        graph.add_node("supervisor", self._node("supervisor", self._supervisor_node))
        graph.add_node("billing", self._node("billing", self._billing_node))
        graph.add_node("tech", self._node("tech", self._tech_node))
        graph.add_node("researcher", self._node("researcher", self._researcher_node))
        graph.add_node("general", self._node("general", self._general_node))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {"billing": "billing", "tech": "tech", "researcher": "researcher", "general": "general", END: END},
        )
        graph.add_edge("billing", "supervisor")
        graph.add_edge("tech", "supervisor")
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("general", "supervisor")
        return graph.compile()

    def _node(self, _name: str, fn: Any) -> Any:
        """Обгортка: додає до `update["trajectory"]` лише нові кроки, залишені
        цим викликом у спільному self.logger — без дублювання (reducer вище
        додатково підстраховує ідемпотентність при повторних візитах).
        """

        def wrapped(state: MASState) -> dict[str, Any]:
            before = len(self.logger.steps)
            update = fn(state)
            delta = self.logger.steps[before:]
            if delta:
                update = {**update, "trajectory": delta}
            return update

        return wrapped

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

    def _billing_node(self, state: MASState) -> dict[str, Any]:
        handoff = state.get("handoff_context")
        goal = handoff["context"] if handoff else self._user_goal(state)
        try:
            plan, results = self.billing_agent.run(goal)
        except SafetyStop as exc:
            message = f"billing: безпечно зупинено ({exc})"
            return {
                "completed": True,
                "handoff_context": None,
                "results": [message],
                "messages": [AIMessage(content=message)],
            }
        answer = "\n".join(results) if results else "Не вдалося опрацювати бюджетний запит."
        return {
            "completed": True,
            "plan": plan,
            "results": results,
            "messages": [AIMessage(content=answer)],
            "handoff_context": None,
        }

    def _tech_node(self, state: MASState) -> dict[str, Any]:
        goal = self._user_goal(state)
        try:
            answer, _used_tools = self.tech_agent.run(goal)
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

    def _researcher_node(self, state: MASState) -> dict[str, Any]:
        goal = self._user_goal(state)
        try:
            answer = self.researcher_agent.run(goal)
        except SafetyStop as exc:
            answer = f"researcher: безпечно зупинено ({exc})"
        return {"completed": True, "results": [f"researcher: {answer}"], "messages": [AIMessage(content=answer)]}

    def _general_node(self, state: MASState) -> dict[str, Any]:
        goal = self._user_goal(state)
        answer = self.general_agent.run(goal)
        return {"completed": True, "results": [f"general: {answer}"], "messages": [AIMessage(content=answer)]}

    # -- публічний вхід --------------------------------------------------------

    def handle_query(self, query: str, thread_id: str = "demo") -> dict[str, Any]:
        initial_state: MASState = {
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
        return self.app.invoke(initial_state)


if __name__ == "__main__":
    from pathlib import Path

    from tools_legacy import configure_storage

    # Демо-запуск — це runtime-сесія, не контрольний артефакт здачі: сховище
    # спрямовуємо у gitignored runtime/, а не в корінь репозиторію.
    runtime_dir = Path(__file__).resolve().parent / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    configure_storage(runtime_dir / "chroma_db", runtime_dir / "bookings.json")

    mas = MASApp(provider="demo", run_id="mas-demo")
    demo_queries = [
        "Скільки коштуватиме подорож до Парижа на 5 днів для двох?",  # billing + реальний replan (>500 EUR)
        "У Львові дощ: перевір погоду і підбери культурні активності",  # tech
        "Які правила відвідування Лувру?",  # researcher
        "Привіт! Що ти вмієш?",  # general
        "Яка погода у Львові і скільки коштуватиме подорож на 3 дні?",  # tech -> handoff -> billing
        "повторюй погоду у Львові",  # tech: контрольована зупинка через детектор циклів
    ]
    for query in demo_queries:
        print(f"\n=== {query} ===")
        final_state = mas.handle_query(query)
        print("current_agent:", final_state.get("current_agent"))
        print("results:", final_state.get("results"))

    from trajectory_logger import save_trajectories

    save_trajectories([mas.logger.as_dict()], runtime_dir / "trajectory_demo_s05.json")
    print(f"\nТраєкторію збережено у {runtime_dir / 'trajectory_demo_s05.json'} (demo-режим).")
