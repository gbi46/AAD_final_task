"""Перевірки evals."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evals


def _patch_runtime_paths(test_case: unittest.TestCase, root: Path) -> None:
    patches = [
        mock.patch.object(evals, "_RUNTIME_DIR", root / "evals"),
        mock.patch.object(evals, "_DB_PATH", root / "evals" / "evals_hitl.db"),
        mock.patch.object(evals, "_KNOWLEDGE_PATH", root / "evals" / "chroma_db"),
        mock.patch.object(evals, "_BOOKING_PATH", root / "evals" / "bookings.json"),
    ]
    for p in patches:
        p.start()
        test_case.addCleanup(p.stop)


class EvalsDemoModeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        _patch_runtime_paths(self, root)
        evals._reset_runtime()
        self.records = await evals._run_all("demo")
        self.by_id = {r.scenario_id: r for r in self.records}

    def test_all_five_scenarios_pass(self) -> None:
        self.assertEqual(len(self.records), 5)
        for record in self.records:
            with self.subTest(scenario=record.scenario_id):
                self.assertEqual(record.status, "pass", record.observed)
                self.assertEqual(record.mode, "demo")
                # Задокументована свідома відсутність зв'язування з трейсом
                # (collect_runs ламає вкладеність — див. docstring evals.py).
                self.assertIsNone(record.trace_id)

    def test_eval01_billing_correct_sum(self) -> None:
        rec = self.by_id["EVAL-01"]
        self.assertEqual(rec.agents_used, ["billing"])
        self.assertIn("calculate_trip_cost", rec.tools_called)
        self.assertIn("312", rec.observed)

    def test_eval02_multi_step_at_least_two_tools(self) -> None:
        rec = self.by_id["EVAL-02"]
        self.assertEqual(rec.agents_used, ["tech"])
        self.assertGreaterEqual(len(rec.tools_called), 2)
        self.assertIn("get_weather", rec.tools_called)
        self.assertIn("find_activities", rec.tools_called)

    def test_eval03_rag_cites_document(self) -> None:
        rec = self.by_id["EVAL-03"]
        self.assertEqual(rec.agents_used, ["researcher"])
        self.assertIn("search_knowledge", rec.tools_called)
        self.assertIn("travel-", rec.observed)

    def test_eval04_real_cross_agent_handoff(self) -> None:
        rec = self.by_id["EVAL-04"]
        self.assertEqual(rec.agents_used, ["tech", "billing"])
        self.assertIn("get_weather", rec.tools_called)
        self.assertIn("calculate_trip_cost", rec.tools_called)

    def test_eval05_hitl_creates_exactly_one_booking(self) -> None:
        rec = self.by_id["EVAL-05"]
        self.assertIn("book_hotel", rec.tools_called)
        self.assertIn("BOOK-", rec.observed)
        # Латентність графа виміряна ОКРЕМО від "очікування людини" .
        self.assertIsNotNone(rec.human_wait_ms)
        self.assertIsNotNone(rec.latency_ms)

    async def test_check_function_can_actually_fail(self) -> None:
        """Контрольна перевірка на "завжди PASS" антипатерн: свідомо хибний
        check має дати status="fail", а не мовчазний pass."""
        outcome = await evals._run_query_scenario(
            "EVAL-01", evals.EVAL01_QUERY, "demo", lambda state: (False, "навмисний fail")
        )
        self.assertEqual(outcome["status"], "fail")


class EvalsLiveModeBlockedTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        _patch_runtime_paths(self, root)
        evals._reset_runtime()
        env_patch = mock.patch.dict(os.environ, {}, clear=False)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop("OPENROUTER_API_KEY", None)

    async def test_missing_key_marks_every_scenario_blocked_not_failed(self) -> None:
        records = await evals._run_all("live")
        self.assertEqual(len(records), 5)
        for record in records:
            with self.subTest(scenario=record.scenario_id):
                self.assertEqual(record.status, "blocked")
                self.assertEqual(record.mode, "live")


class PassRateCalculationTests(unittest.TestCase):
    def test_pass_rate_excludes_blocked_from_denominator(self) -> None:
        # 2 pass, 1 fail, 2 blocked -> pass_rate рахується як 2/3 (executed),
        # Blocked-сценарії показуються окремо й не входять у знаменник.
        statuses = ["pass", "pass", "fail", "blocked", "blocked"]
        executed = [s for s in statuses if s in ("pass", "fail")]
        passed = [s for s in executed if s == "pass"]
        pass_rate = len(passed) / len(executed)
        self.assertAlmostEqual(pass_rate, 2 / 3)


if __name__ == "__main__":
    unittest.main()
