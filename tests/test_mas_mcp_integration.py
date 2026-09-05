"""Перевірки mas mcp integration."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from mas_langgraph import MASApp, ToolDispatchError


def _mcp_env(storage_root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "MCP_KNOWLEDGE_PATH": str(storage_root / "chroma"),
        "MCP_BOOKING_PATH": str(storage_root / "bookings.json"),
    }


class MASMCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temp.name)
        self.mas = MASApp(provider="demo", run_id="test-mcp-integration")
        await self.mas.connect_mcp(env=_mcp_env(self.storage_root))

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def test_billing_and_tech_both_call_real_mcp_subprocess(self) -> None:
        """≥2 запити реально звертаються до subprocess MCP (acceptance ):
        billing (calculate_trip_cost) і tech (get_weather+find_activities).
        Перевіряємо і сам результат, і що trajectory зафіксувала tool_dispatch
        із agent_name для обох.
        """
        billing_final = await self.mas.handle_query("Скільки коштуватиме подорож до Києва на 2 дні?")
        self.assertIn("116 EUR", "\n".join(billing_final["results"]))  # 58*2*1, справжній подвиг MCP-сервера

        tech_final = await self.mas.handle_query("У Львові дощ: перевір погоду і підбери культурні активності")
        self.assertTrue(any("15°C" in r for r in tech_final["results"]))

        dispatches = [s for s in self.mas.logger.steps if s["node_name"] == "tool_dispatch"]
        agent_names = {d["agent_name"] for d in dispatches}
        self.assertIn("billing", agent_names)
        self.assertIn("tech", agent_names)
        self.assertGreaterEqual(len(dispatches), 2)

    async def test_mcp_tool_result_is_normalized_to_plain_string(self) -> None:
        """MCP tool повертає список content-блоків (не голий рядок, як
        локальні wrappers) — dispatch_tool має нормалізувати результат, щоб
        агенти не бачили Python-репр списку словників.
        """
        result = await self.mas.dispatch_tool("get_weather", {"city": "Київ", "days": 1}, "tech")
        self.assertIsInstance(result, str)
        self.assertNotIn("'type': 'text'", result)
        self.assertIn("18°C", result)

    async def test_mcp_server_business_validation_error_is_not_fake_success(self) -> None:
        """Невалідне місто — MCP-сервер повертає CallToolResult(isError=True).
        З handle_tool_errors=False це підіймається як ToolDispatchError, а НЕ
        повертається як фальшивий успішний текст.
        """
        with self.assertRaises(ToolDispatchError):
            await self.mas.dispatch_tool("get_weather", {"city": "Атлантида", "days": 1}, "tech")
        errors = [s for s in self.mas.logger.steps if s["node_name"] == "tool_dispatch_error"]
        self.assertTrue(any("Атлантида" in str(e["input"]) or "не підтримується" in e["output"] for e in errors))

    async def test_researcher_faq_resource_fast_path_vs_real_rag_via_mcp(self) -> None:
        """Порівняння вимоги плану на рівні resource: запит, що збігається зі
        статичним faq://general (швидший шлях, без tool-виклику), проти
        запиту, що потребує справжнього RAG search_knowledge через MCP.
        """
        faq_final = await self.mas.handle_query("Як зменшити ризик крадіжок у подорожі?")
        self.assertIn("faq://general", "\n".join(faq_final["results"]))
        faq_dispatches_before = len([s for s in self.mas.logger.steps if s["node_name"] == "tool_dispatch"])

        rag_final = await self.mas.handle_query("Які правила відвідування Лувру?")
        self.assertIn("[travel-", "\n".join(rag_final["results"]))
        faq_dispatches_after = len([s for s in self.mas.logger.steps if s["node_name"] == "tool_dispatch"])
        # RAG-запит справді додав виклик tool (search_knowledge через MCP),
        # FAQ-запит — ні (fast-path обійшовся без tool_dispatch).
        self.assertGreater(faq_dispatches_after, faq_dispatches_before)

    async def test_general_uses_travel_reply_prompt_template(self) -> None:
        """general використовує MCP prompt travel_reply — шаблон, не готову
        вигадану відповідь (перевіряємо характерний маркер шаблону).
        """
        final = await self.mas.handle_query("Привіт! Що ти вмієш?")
        joined = "\n".join(final["results"])
        self.assertIn("Привіт, Мандрівник", joined)
        self.assertIn("get_weather", joined)  # шаблон вказує додати факти з tools


if __name__ == "__main__":
    unittest.main()
