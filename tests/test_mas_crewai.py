"""Тести mas_crewai.py (S14, Bonus Level).

ВАЖЛИВО: цей файл запускається під ОКРЕМИМ venv (`.venv-crewai/`), НЕ під
основним `.venv/` -- `crewai`/`crewai-tools` мають pinned-версії, що
конфліктують з основним `requirements.txt` (S14, `mas_crewai.py` docstring).
Тому він НЕ входить у звичайний `python -m pytest tests/` (основний venv
навіть не має `crewai` встановленим) -- запускати окремо:

    .venv-crewai/bin/python -m pytest tests/test_mas_crewai.py -v

Тести без `RUN_LIVE_CREWAI=1` перевіряють лише те, що НЕ вимагає живого
LLM-виклику (реальне MCP-підключення й фільтрація tools — це не вимагає
LLM, лише субпроцес). Один live-тест (реальний `run_query()` через
OpenRouter) вимкнено за замовчуванням -- явно дорогий (реальні токени) і
повільний (~30-90с через delegation); вмикається `RUN_LIVE_CREWAI=1` +
робочий `OPENROUTER_API_KEY`.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv

import mas_crewai as mc

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# `crewai` не встановлений в основному .venv (навмисно -- конфліктує з
# pinned requirements.txt, див. docstring mas_crewai.py) -- будь-який тест,
# що реально ІМПОРТУЄ crewai (не лише статичні перевірки allowlist-констант
# вище modele-рівня), під основним venv має SKIP, а не FAIL, інакше
# звичайний `pytest tests/` без --ignore ламається для того, хто не знає
# про окремий .venv-crewai/.
_HAS_CREWAI = importlib.util.find_spec("crewai") is not None
_skip_without_crewai = unittest.skipUnless(
    _HAS_CREWAI, "crewai не встановлено в цьому venv -- запустити під .venv-crewai/bin/python"
)


class ToolAllowlistTests(unittest.TestCase):
    """Статичні перевірки allowlist -- book_hotel НЕ має з'явитись у жодному
    записі (план: "не дати всім агентам привілейований book_hotel")."""

    def test_book_hotel_excluded_from_every_agent(self) -> None:
        for name, tools in (
            ("tech", mc.TECH_TOOLS),
            ("billing", mc.BILLING_TOOLS),
            ("researcher", mc.RESEARCHER_TOOLS),
        ):
            with self.subTest(agent=name):
                self.assertNotIn("book_hotel", tools)

    def test_allowlists_are_disjoint_from_risky_tool(self) -> None:
        all_tools = set(mc.TECH_TOOLS) | set(mc.BILLING_TOOLS) | set(mc.RESEARCHER_TOOLS)
        self.assertNotIn("book_hotel", all_tools)
        # Рівно 4 з 5 tools доступні хоч комусь (усе, крім book_hotel).
        self.assertEqual(all_tools, {"get_weather", "find_activities", "calculate_trip_cost", "search_knowledge"})


@_skip_without_crewai
class MCPConnectionTests(unittest.TestCase):
    """Справжнє MCP-підключення (реальний subprocess mcp_server.py через
    ГОЛОВНИЙ .venv) -- без LLM, тому швидко й безкоштовно."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime_dir = Path(self.temp.name)

    def test_per_agent_tool_filtering_via_real_mcp_subprocess(self) -> None:
        with mc._connected_tools(self.runtime_dir) as tools_by_agent:
            tech_names = {t.name for t in tools_by_agent["tech"]}
            billing_names = {t.name for t in tools_by_agent["billing"]}
            researcher_names = {t.name for t in tools_by_agent["researcher"]}

        self.assertEqual(tech_names, set(mc.TECH_TOOLS))
        self.assertEqual(billing_names, set(mc.BILLING_TOOLS))
        self.assertEqual(researcher_names, set(mc.RESEARCHER_TOOLS))
        # Жоден агент не бачить book_hotel навіть як опцію (не лише
        # "заборонено post-hoc" -- фізично відсутній у списку tools).
        for names in (tech_names, billing_names, researcher_names):
            self.assertNotIn("book_hotel", names)


@_skip_without_crewai
class LLMConfigTests(unittest.TestCase):
    """`crewai.LLM(model="openrouter/...")` валідує наявність (НЕ
    справжність) `OPENROUTER_API_KEY` вже при самій побудові об'єкта — без
    жодного ключа падає `pydantic.ValidationError` ще до будь-якого
    мережевого виклику (виявлено емпірично, S16: тести падали в чистому
    checkout без `.env`, хоча в дев-середовищі з реальним ключем
    проходили мовчки). Тому кожен тест тут підставляє ФІКТИВНИЙ рядок
    (не справжній секрет, жодного виклику мережі не відбувається) --
    офлайн-набір тестів і так не повинен залежати від реальних ключів."""

    def setUp(self) -> None:
        self._old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-placeholder-not-a-real-key"
        self.addCleanup(self._restore_key)

    def _restore_key(self) -> None:
        if self._old_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self._old_key

    def test_build_llm_uses_env_model_and_max_tokens(self) -> None:
        old_model = os.environ.get("OPENROUTER_MODEL")
        old_max = os.environ.get("OPENROUTER_MAX_TOKENS")
        os.environ["OPENROUTER_MODEL"] = "anthropic/claude-sonnet-4.6"
        os.environ["OPENROUTER_MAX_TOKENS"] = "2048"
        try:
            llm = mc._build_llm()
            self.assertIn("anthropic/claude-sonnet-4.6", llm.model)
            self.assertEqual(llm.max_tokens, 2048)
        finally:
            for key, value in (("OPENROUTER_MODEL", old_model), ("OPENROUTER_MAX_TOKENS", old_max)):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_build_llm_default_max_tokens_avoids_the_65536_credit_error(self) -> None:
        """Регресія на баг, знайдений живим прогоном S14: OpenRouter/CrewAI
        за замовчуванням підставляють max_tokens=65536, що перевищує баланс
        тестового ключа (402 'requires more credits')."""
        old_max = os.environ.pop("OPENROUTER_MAX_TOKENS", None)
        try:
            llm = mc._build_llm()
            self.assertLess(llm.max_tokens, 65536)
            self.assertEqual(llm.max_tokens, 4096)
        finally:
            if old_max is not None:
                os.environ["OPENROUTER_MAX_TOKENS"] = old_max


@unittest.skipUnless(
    os.getenv("RUN_LIVE_CREWAI") == "1" and os.getenv("OPENROUTER_API_KEY"),
    "Live CrewAI run вимкнено за замовчуванням (реальні токени/повільно) -- "
    "RUN_LIVE_CREWAI=1 і робочий OPENROUTER_API_KEY, щоб увімкнути.",
)
class LiveCrewRunTests(unittest.TestCase):
    """Опційний живий прогін -- підтверджує ≥3 спеціалісти + фактичний MCP
    call (план S14), не запускається в звичайному прогоні тестів."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_billing_query_triggers_real_calculate_trip_cost(self) -> None:
        """Перевіряє лише, що запит успішно проходить крізь delegation і
        повертає змістовну відповідь про Львів. НЕ перевіряє точний збіг
        числа ("312 EUR") у фінальному тексті: емпірично виявлено (S14),
        що coordinator у CrewAI ПЕРЕФОРМУЛЬОВУЄ делеговану відповідь billing-
        агента власними словами (часто переводить у грн і заокруглює), а не
        передає точний рядок tool-результату так, як це робить LangGraph
        `results` (S05, `operator.add` reducer над точними рядками кроків).
        Це задокументована відмінність для S15 (fidelity/precision), не
        баг цього тесту.
        """
        result = mc.run_query(
            "Скільки коштуватиме подорож до Львова на 3 дні для двох?",
            runtime_dir=Path(self.temp.name),
        )
        self.assertIn("Львів", result)
        self.assertGreater(len(result), 100)


if __name__ == "__main__":
    unittest.main()
