"""Тести MAS-графа: маршрутизація, реальний replan, handoff, ліміти (S05).

Усі тести працюють у demo-режимі (без LLM, без API-ключа) — `DemoSupervisorRouter`
проходить той самий граф, що й live-router, лише інше джерело `RouteDecision`.
Сховище (ChromaDB + booking JSON) ізольоване через `tools_legacy.configure_storage`
у tempdir на кожен тест.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

from tools_legacy import configure_storage

import mas_langgraph as mas_module
from mas_langgraph import MASApp, MAX_HANDOFFS


class MASRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-routing")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_billing_branch(self) -> None:
        final = self.mas.handle_query("Скільки коштуватиме подорож до Києва на 2 дні?")
        self.assertEqual(final["current_agent"], "billing")
        self.assertTrue(final["completed"])
        self.assertTrue(any("calculate_trip_cost" in r for r in final["results"]))

    def test_tech_branch(self) -> None:
        final = self.mas.handle_query("У Львові дощ: перевір погоду і підбери культурні активності")
        self.assertEqual(final["current_agent"], "tech")
        self.assertTrue(final["completed"])
        self.assertTrue(any("15°C" in r or "дощ" in r for r in final["results"]))

    def test_researcher_branch_with_document_ids(self) -> None:
        final = self.mas.handle_query("Які правила відвідування Лувру?")
        self.assertEqual(final["current_agent"], "researcher")
        self.assertTrue(final["completed"])
        joined = "\n".join(final["results"])
        self.assertIn("Лувр", joined)
        self.assertIn("[travel-", joined)  # RAG-відповідь цитує ID документів

    def test_general_branch_no_risky_calls(self) -> None:
        final = self.mas.handle_query("Привіт! Що ти вмієш?")
        self.assertEqual(final["current_agent"], "general")
        self.assertTrue(final["completed"])
        # general не має жодного ToolMessage — жодного tool-виклику не було.
        from langchain_core.messages import ToolMessage

        self.assertFalse(any(isinstance(m, ToolMessage) for m in final["messages"]))

    def test_arithmetic_query_does_not_need_rag_comparison(self) -> None:
        """Порівняння: довідковий запит (researcher, RAG потрібен) проти
        арифметичного (billing, RAG НЕ потрібен) — різні гілки, жодного
        виклику search_knowledge у billing-результатах.
        """
        rag_final = self.mas.handle_query("Які правила відвідування Лувру?")
        arithmetic_final = self.mas.handle_query("Скільки коштуватиме подорож до Одеси на 2 дні?")
        self.assertEqual(rag_final["current_agent"], "researcher")
        self.assertEqual(arithmetic_final["current_agent"], "billing")
        self.assertFalse(any("search_knowledge" in r for r in arithmetic_final["results"]))


class MASReplanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-replan")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_high_budget_triggers_real_replan(self) -> None:
        """Париж, 5 днів, 2 особи -> 1450 EUR > поріг 500 EUR -> replanner
        реально додає крок search_knowledge з порадами економії (не лише
        позначає рішення - план фізично зростає з 1 до 2 кроків).
        """
        final = self.mas.handle_query("Скільки коштуватиме подорож до Парижа на 5 днів для двох?")
        self.assertEqual(len(final["plan"]), 2)
        self.assertIn("поради для економії", final["plan"][1])
        self.assertEqual(len(final["results"]), 2)
        self.assertIn("1450 EUR", final["results"][0])

    def test_low_budget_does_not_trigger_replan(self) -> None:
        """Київ, 2 дні, 1 особа -> низький бюджет -> план лишається з 1 кроку."""
        final = self.mas.handle_query("Скільки коштуватиме подорож до Києва на 2 дні?")
        self.assertEqual(len(final["plan"]), 1)
        self.assertEqual(len(final["results"]), 1)


class MASHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-handoff")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_tech_to_billing_real_handoff(self) -> None:
        final = self.mas.handle_query("Яка погода у Львові і скільки коштуватиме подорож на 3 дні?")
        # Завершує саме billing (після реального handoff від tech), не tech.
        self.assertEqual(final["current_agent"], "billing")
        self.assertEqual(final["handoff_count"], 1)
        self.assertTrue(final["completed"])
        # Обидва внески — і tech (погода), і billing (бюджет) — присутні в results.
        joined = "\n".join(final["results"])
        self.assertIn("tech:", joined)
        self.assertIn("calculate_trip_cost", joined)

    def test_handoff_count_never_exceeds_limit(self) -> None:
        final = self.mas.handle_query("Яка погода у Львові і скільки коштуватиме подорож на 3 дні?")
        self.assertLessEqual(final["handoff_count"], MAX_HANDOFFS)


class MASSafetyLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")
        self.mas = MASApp(provider="demo", run_id="test-safety")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_repeat_cycle_stops_graph_safely(self) -> None:
        """tech: детектор повторюваних викликів зупиняє граф контрольовано —
        MASApp.handle_query повертає результат (не крашиться, не зависає),
        completed=True з поясненням safety-зупинки.
        """
        final = self.mas.handle_query("повторюй погоду у Львові")
        self.assertEqual(final["current_agent"], "tech")
        self.assertTrue(final["completed"])
        joined = "\n".join(final["results"])
        self.assertIn("безпечно зупинено", joined)
        self.assertIn("цикл", joined)

    def test_global_supervisor_visit_limit_stops_graph(self) -> None:
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
        final = self.mas.app.invoke(initial_state)
        self.assertTrue(final["completed"])
        self.assertIn("глобальний ліміт", "\n".join(final["results"]))


if __name__ == "__main__":
    unittest.main()
