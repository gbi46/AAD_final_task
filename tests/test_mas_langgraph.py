"""Тести MAS-графа: маршрутизація, реальний replan, handoff, ліміти (S05+S06).

Усі тести працюють у demo-режимі (без LLM, без API-ключа) — `DemoSupervisorRouter`
проходить той самий граф, що й live-router, лише інше джерело `RouteDecision`.
Виконання tools тут — локальні wrappers з `tools_legacy.py` (MCP не
підключено, `connect_mcp()` не викликається); MCP-специфічну поведінку
(dispatch_tool, нормалізація content-блоків, isError-обробка) перевіряє
окремий сценарій у `mcp_server.py`/`test_mcp_server.py` та ручна demo-
перевірка S06 (див. STEP_LOG.md).

`MASApp.handle_query`/`.run()` — асинхронні (S06: реальний dispatch_tool
може викликати `await tool.ainvoke(...)`), тому тести успадковують
`unittest.IsolatedAsyncioTestCase`, а не звичайний `TestCase`. Сховище
(ChromaDB + booking JSON) ізольоване через `tools_legacy.configure_storage`
у tempdir на кожен тест.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from tools_legacy import configure_storage

import mas_langgraph as mas_module
from mas_langgraph import MASApp, MAX_HANDOFFS


class MASRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-routing")

    def tearDown(self) -> None:
        self.temp.cleanup()

    async def test_billing_branch(self) -> None:
        final = await self.mas.handle_query("Скільки коштуватиме подорож до Києва на 2 дні?")
        self.assertEqual(final["current_agent"], "billing")
        self.assertTrue(final["completed"])
        self.assertTrue(any("calculate_trip_cost" in r for r in final["results"]))

    async def test_tech_branch(self) -> None:
        final = await self.mas.handle_query("У Львові дощ: перевір погоду і підбери культурні активності")
        self.assertEqual(final["current_agent"], "tech")
        self.assertTrue(final["completed"])
        self.assertTrue(any("15°C" in r or "дощ" in r for r in final["results"]))

    async def test_researcher_branch_with_document_ids(self) -> None:
        final = await self.mas.handle_query("Які правила відвідування Лувру?")
        self.assertEqual(final["current_agent"], "researcher")
        self.assertTrue(final["completed"])
        joined = "\n".join(final["results"])
        self.assertIn("Лувр", joined)
        self.assertIn("[travel-", joined)  # RAG-відповідь цитує ID документів

    async def test_general_branch_no_risky_calls(self) -> None:
        final = await self.mas.handle_query("Привіт! Що ти вмієш?")
        self.assertEqual(final["current_agent"], "general")
        self.assertTrue(final["completed"])
        # general не має жодного ToolMessage — жодного tool-виклику не було.
        self.assertFalse(any(isinstance(m, ToolMessage) for m in final["messages"]))

    async def test_arithmetic_query_does_not_need_rag_comparison(self) -> None:
        """Порівняння: довідковий запит (researcher, RAG потрібен) проти
        арифметичного (billing, RAG НЕ потрібен) — різні гілки, жодного
        виклику search_knowledge у billing-результатах.
        """
        rag_final = await self.mas.handle_query("Які правила відвідування Лувру?")
        arithmetic_final = await self.mas.handle_query("Скільки коштуватиме подорож до Одеси на 2 дні?")
        self.assertEqual(rag_final["current_agent"], "researcher")
        self.assertEqual(arithmetic_final["current_agent"], "billing")
        self.assertFalse(any("search_knowledge" in r for r in arithmetic_final["results"]))


class MASReplanTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-replan")

    def tearDown(self) -> None:
        self.temp.cleanup()

    async def test_high_budget_triggers_real_replan(self) -> None:
        """Париж, 5 днів, 2 особи -> 1450 EUR > поріг 500 EUR -> replanner
        реально додає крок search_knowledge з порадами економії (не лише
        позначає рішення - план фізично зростає з 1 до 2 кроків).
        """
        final = await self.mas.handle_query("Скільки коштуватиме подорож до Парижа на 5 днів для двох?")
        self.assertEqual(len(final["plan"]), 2)
        self.assertIn("поради для економії", final["plan"][1])
        self.assertEqual(len(final["results"]), 2)
        self.assertIn("1450 EUR", final["results"][0])

    async def test_low_budget_does_not_trigger_replan(self) -> None:
        """Київ, 2 дні, 1 особа -> низький бюджет -> план лишається з 1 кроку."""
        final = await self.mas.handle_query("Скільки коштуватиме подорож до Києва на 2 дні?")
        self.assertEqual(len(final["plan"]), 1)
        self.assertEqual(len(final["results"]), 1)


class MASHandoffTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-handoff")

    def tearDown(self) -> None:
        self.temp.cleanup()

    async def test_tech_to_billing_real_handoff(self) -> None:
        final = await self.mas.handle_query("Яка погода у Львові і скільки коштуватиме подорож на 3 дні?")
        # Завершує саме billing (після реального handoff від tech), не tech.
        self.assertEqual(final["current_agent"], "billing")
        self.assertEqual(final["handoff_count"], 1)
        self.assertTrue(final["completed"])
        # Обидва внески — і tech (погода), і billing (бюджет) — присутні в results.
        joined = "\n".join(final["results"])
        self.assertIn("tech:", joined)
        self.assertIn("calculate_trip_cost", joined)

    async def test_handoff_count_never_exceeds_limit(self) -> None:
        final = await self.mas.handle_query("Яка погода у Львові і скільки коштуватиме подорож на 3 дні?")
        self.assertLessEqual(final["handoff_count"], MAX_HANDOFFS)


class MASSafetyLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-safety")

    def tearDown(self) -> None:
        self.temp.cleanup()

    async def test_repeat_cycle_stops_graph_safely(self) -> None:
        """tech: детектор повторюваних викликів зупиняє граф контрольовано —
        MASApp.handle_query повертає результат (не крашиться, не зависає),
        completed=True з поясненням safety-зупинки.
        """
        final = await self.mas.handle_query("повторюй погоду у Львові")
        self.assertEqual(final["current_agent"], "tech")
        self.assertTrue(final["completed"])
        joined = "\n".join(final["results"])
        self.assertIn("безпечно зупинено", joined)
        self.assertIn("цикл", joined)

    async def test_global_supervisor_visit_limit_stops_graph(self) -> None:
        """Стартуємо invoke зі стану, де step_count вже дорівнює
        MAX_SUPERVISOR_VISITS (як після кількох попередніх ітерацій) —
        наступний візит supervisor має перевищити ліміт і безпечно
        завершити граф, а не покладатися лише на handoff-логіку чи
        нормальну маршрутизацію.
        """
        initial_state = {
            "messages": [HumanMessage(content="Привіт!")],
            "current_agent": "general",
            "plan": [],
            "current_step": 0,
            "results": [],
            "step_count": mas_module.MAX_SUPERVISOR_VISITS,  # наступний візит supervisor вже перевищить ліміт
            "trajectory": [],
            "completed": False,
            "pending_approval": None,
            "thread_id": "limit-test",
            "agent_step_counts": {},
            "handoff_context": None,
            "handoff_count": 0,
        }
        final = await self.mas.app.ainvoke(initial_state)
        self.assertTrue(final["completed"])
        self.assertIn("глобальний ліміт", "\n".join(final["results"]))


class MASDispatchToolTests(unittest.IsolatedAsyncioTestCase):
    """Прямі тести `MASApp.dispatch_tool` (S06): єдиний шлях виконання tools."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-dispatch")

    def tearDown(self) -> None:
        self.temp.cleanup()

    async def test_dispatch_unknown_tool_raises_and_logs(self) -> None:
        with self.assertRaises(mas_module.ToolDispatchError):
            await self.mas.dispatch_tool("no_such_tool", {}, "tech")
        errors = [s for s in self.mas.logger.steps if s["node_name"] == "tool_dispatch_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["agent_name"], "tech")

    async def test_dispatch_known_tool_records_success_with_agent_name(self) -> None:
        result = await self.mas.dispatch_tool("get_weather", {"city": "Київ", "days": 1}, "tech")
        self.assertIn("18°C", result)
        successes = [s for s in self.mas.logger.steps if s["node_name"] == "tool_dispatch"]
        self.assertEqual(len(successes), 1)
        self.assertEqual(successes[0]["agent_name"], "tech")
        self.assertEqual(successes[0]["tool_calls"][0]["name"], "get_weather")
        self.assertIn("id", successes[0]["tool_calls"][0])  # tool_call_id збережено


if __name__ == "__main__":
    unittest.main()
