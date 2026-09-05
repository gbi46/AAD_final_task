"""Перевірки red team."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import guardrails
import mas_langgraph
import red_team


def _patch_runtime_paths(test_case: unittest.TestCase, root: Path) -> None:
    patches = [
        mock.patch.object(red_team, "_RUNTIME_DIR", root / "red_team"),
        mock.patch.object(red_team, "_DB_PATH", root / "red_team" / "red_team_hitl.db"),
        mock.patch.object(red_team, "_KNOWLEDGE_PATH", root / "red_team" / "chroma_db"),
        mock.patch.object(red_team, "_BOOKING_PATH", root / "red_team" / "bookings.json"),
    ]
    for p in patches:
        p.start()
        test_case.addCleanup(p.stop)


class RedTeamDemoModeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        _patch_runtime_paths(self, Path(self.temp.name))
        self.records = await red_team._run_all("demo")
        self.by_id = {r.test_id: r for r in self.records}

    def test_all_seven_scenarios_pass(self) -> None:
        self.assertEqual(len(self.records), 7)
        for record in self.records:
            with self.subTest(scenario=record.test_id):
                self.assertEqual(record.status, "pass", record.observed)
                self.assertEqual(record.mode, "demo")

    def test_rt01_zero_downstream_calls(self) -> None:
        rec = self.by_id["RT-01"]
        self.assertEqual(rec.blocked_by, "input_guardrail")
        self.assertEqual(rec.side_effect_count, 0)
        self.assertFalse(rec.controlled)

    def test_rt02_raw_pii_actually_present_before_redaction(self) -> None:
        """сама наявність [REDACTED] без підтвердженого
        сирого PII на вході -- НЕ доказ. Перевіряємо, що red_team.py
        реально підтверджує обидва боки."""
        rec = self.by_id["RT-02"]
        self.assertIn("raw_pii_in_tool_result=True", rec.observed)
        self.assertIn("raw_leaked_in_response=False", rec.observed)
        self.assertIn("raw_leaked_in_trajectory=False", rec.observed)
        self.assertNotIn("ivan.test@example.com", rec.observed.split("results=")[-1])

    def test_rt03_is_marked_controlled(self) -> None:
        rec = self.by_id["RT-03"]
        self.assertTrue(rec.controlled)
        self.assertEqual(rec.blocked_by, "tool_guardrail")
        self.assertEqual(rec.side_effect_count, 0)

    def test_rt04_no_booking_before_resume(self) -> None:
        rec = self.by_id["RT-04"]
        self.assertEqual(rec.blocked_by, "approval_gate/interrupt")
        self.assertEqual(rec.side_effect_count, 0)

    def test_rt06_is_marked_controlled_and_doc_was_retrieved(self) -> None:
        rec = self.by_id["RT-06"]
        self.assertTrue(rec.controlled)
        self.assertIn("doc_retrieved=True", rec.observed)
        self.assertEqual(rec.side_effect_count, 0)

    def test_rt07_session_isolation(self) -> None:
        rec = self.by_id["RT-07"]
        self.assertEqual(rec.blocked_by, "rate_limit_guardrail")


class RedTeamNegativeControlTests(unittest.IsolatedAsyncioTestCase):
    """Захист від rubber-stamp "завжди pass": якщо guardrail дійсно
    обійдено, red_team.py має це виявити."""

    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        _patch_runtime_paths(self, Path(self.temp.name))

        original_guard = mas_langgraph.tool_guardrail
        mas_langgraph.tool_guardrail = lambda agent_name, tool_name: True  # симуляція bypass

        def _restore() -> None:
            mas_langgraph.tool_guardrail = original_guard

        self.addCleanup(_restore)

    async def test_rt03_reports_fail_when_guardrail_is_actually_bypassed(self) -> None:
        red_team._reset_runtime()
        record = await red_team._run_rt03("demo")
        self.assertEqual(record.status, "fail")
        self.assertIsNone(record.blocked_by)


if __name__ == "__main__":
    unittest.main()
