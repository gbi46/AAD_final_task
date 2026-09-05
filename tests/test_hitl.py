"""Тести HITL (Human-in-the-Loop) для book_hotel (S09).

Покриває ADV-07..ADV-11 з codex_review/REQUIREMENTS.md:
`interrupt()`/`Command(resume=...)` з файловим `AsyncSqliteSaver`,
approve/reject/edit/invalid-edit, restart+replay без дублю, mixed
safe/risky batch, researcher не може book_hotel навіть через
approval_gate. Перевіряється вміст `bookings.json` до/після, а не лише
текст відповіді.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from hitl import is_valid_decision
from mas_langgraph import MASApp, ToolDispatchError
from tools_legacy import configure_storage

_HITL_SCRIPT = Path(__file__).resolve().parent.parent / "hitl.py"
_BOOKING_QUERY = "Забронюй готель на 15 вересня 2026 на 2 ночі"


class IsValidDecisionTests(unittest.TestCase):
    def test_approve_reject_edit_are_valid(self) -> None:
        for action in ("approve", "reject", "edit"):
            self.assertTrue(is_valid_decision({"action": action}))

    def test_boolean_approved_is_not_a_valid_decision(self) -> None:
        """Ключова вимога плану: boolean approved=True, який LLM міг би сам
        передати, НЕ є людським підтвердженням."""
        self.assertFalse(is_valid_decision({"approved": True}))
        self.assertFalse(is_valid_decision(True))
        self.assertFalse(is_valid_decision("approve"))
        self.assertFalse(is_valid_decision({"action": "delete_everything"}))
        self.assertFalse(is_valid_decision(None))


class MASApprovalGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        configure_storage(self.root / "chroma", self.root / "bookings.json")
        self._saver_cm = AsyncSqliteSaver.from_conn_string(str(self.root / "state.db"))
        self.saver = await self._saver_cm.__aenter__()
        self.mas = MASApp(provider="demo", run_id="test-hitl", checkpointer=self.saver)

    async def asyncTearDown(self) -> None:
        await self._saver_cm.__aexit__(None, None, None)
        self.temp.cleanup()

    @property
    def booking_path(self) -> Path:
        return self.root / "bookings.json"

    async def _pause(self, thread_id: str, query: str = _BOOKING_QUERY) -> dict:
        state = await self.mas.handle_query(query, thread_id=thread_id)
        interrupts = state.get("__interrupt__")
        self.assertTrue(interrupts, "Граф мав зупинитися на interrupt()")
        return interrupts[0].value

    async def test_interrupt_payload_has_required_fields(self) -> None:
        payload = await self._pause("t-payload")
        self.assertEqual(payload["tool"], "book_hotel")
        self.assertEqual(payload["agent_name"], "billing")
        self.assertIn("args", payload)
        self.assertIn("consequences", payload)
        self.assertIn("operation_id", payload)
        self.assertFalse(self.booking_path.exists(), "До рішення людини жодного booking не мало бути")

    async def test_approve_executes_tool_exactly_once(self) -> None:
        await self._pause("t-approve")
        final = await self.mas.app.ainvoke(Command(resume={"action": "approve"}), config=self.mas.config("t-approve"))
        self.assertTrue(final["completed"])
        self.assertTrue(any("BOOK-001" in r for r in final["results"]))
        stored = json.loads(self.booking_path.read_text(encoding="utf-8"))
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["status"], "confirmed")

    async def test_reject_no_side_effect_reason_saved(self) -> None:
        await self._pause("t-reject")
        final = await self.mas.app.ainvoke(
            Command(resume={"action": "reject", "reason": "Клієнт передумав"}), config=self.mas.config("t-reject")
        )
        joined = "\n".join(final["results"])
        self.assertIn("Клієнт передумав", joined)
        self.assertIn("не виконувався", joined)
        self.assertFalse(self.booking_path.exists())

    async def test_edit_valid_creates_modified_record(self) -> None:
        await self._pause("t-edit")
        final = await self.mas.app.ainvoke(
            Command(resume={"action": "edit", "args": {"nights": 5, "total_cost": 700.0}}),
            config=self.mas.config("t-edit"),
        )
        self.assertTrue(any("BOOK-001" in r for r in final["results"]))
        stored = json.loads(self.booking_path.read_text(encoding="utf-8"))
        self.assertEqual(stored[0]["nights"], 5)
        self.assertEqual(stored[0]["total_cost"], 700.0)

    async def test_edit_invalid_does_not_create_record(self) -> None:
        await self._pause("t-invalid-edit")
        final = await self.mas.app.ainvoke(
            Command(resume={"action": "edit", "args": {"nights": -1}}), config=self.mas.config("t-invalid-edit")
        )
        self.assertIn("невалідне", "\n".join(final["results"]))
        self.assertFalse(self.booking_path.exists())

    async def test_invalid_action_denied_not_silent_approve(self) -> None:
        await self._pause("t-invalid-action")
        final = await self.mas.app.ainvoke(
            Command(resume={"action": "definitely_not_a_real_action"}), config=self.mas.config("t-invalid-action")
        )
        self.assertIn("Некоректне", "\n".join(final["results"]))
        self.assertFalse(self.booking_path.exists())

    async def test_boolean_approved_true_denied_not_silent_approve(self) -> None:
        await self._pause("t-bool-approve")
        final = await self.mas.app.ainvoke(Command(resume={"approved": True}), config=self.mas.config("t-bool-approve"))
        self.assertIn("Некоректне", "\n".join(final["results"]))
        self.assertFalse(self.booking_path.exists())

    async def test_repeat_resume_does_not_duplicate(self) -> None:
        """ADV-11: replay після resume/restart не створює дубль."""
        await self._pause("t-replay")
        first = await self.mas.app.ainvoke(Command(resume={"action": "approve"}), config=self.mas.config("t-replay"))
        second = await self.mas.app.ainvoke(None, config=self.mas.config("t-replay"))
        self.assertEqual(first["results"], second["results"])
        stored = json.loads(self.booking_path.read_text(encoding="utf-8"))
        self.assertEqual(len(stored), 1)

    async def test_mixed_safe_and_risky_batch(self) -> None:
        """Один план поєднує безпечний крок (calculate_trip_cost —
        виконується одразу) і ризиковий (book_hotel — зупиняється на
        interrupt); після approve обидва результати присутні."""
        state = await self.mas.handle_query(_BOOKING_QUERY, thread_id="t-mixed")
        self.assertEqual(len(state.get("plan", [])), 2)
        self.assertTrue(any("calculate_trip_cost" in r for r in state["results"]))
        self.assertTrue(state.get("__interrupt__"))

        final = await self.mas.app.ainvoke(Command(resume={"action": "approve"}), config=self.mas.config("t-mixed"))
        self.assertTrue(any("calculate_trip_cost" in r for r in final["results"]))
        self.assertTrue(any("book_hotel" in r for r in final["results"]))

    async def test_researcher_cannot_book_hotel_even_via_dispatch(self) -> None:
        with self.assertRaises(ToolDispatchError):
            await self.mas.dispatch_tool(
                "book_hotel",
                {"hotel_name": "X", "check_in": "2026-09-15", "nights": 1, "total_cost": 100.0},
                "researcher",
            )
        self.assertFalse(self.booking_path.exists())

    async def test_different_thread_ids_get_independent_approvals(self) -> None:
        """Дві незалежні сесії бронювання не заважають одна одній: session-a
        відхиляє (жодного booking), session-b підтверджує (рівно один)."""
        await self._pause("t-session-a")
        await self._pause("t-session-b")
        final_a = await self.mas.app.ainvoke(
            Command(resume={"action": "reject", "reason": "A"}), config=self.mas.config("t-session-a")
        )
        final_b = await self.mas.app.ainvoke(Command(resume={"action": "approve"}), config=self.mas.config("t-session-b"))
        self.assertIn("не виконувався", "\n".join(final_a["results"]))
        self.assertTrue(any("BOOK-001" in r for r in final_b["results"]))
        stored = json.loads(self.booking_path.read_text(encoding="utf-8"))
        self.assertEqual(len(stored), 1)  # лише від session-b; session-a нічого не створила


class HitlCLICrossProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp.name)
        self.env = {
            **os.environ,
            "HITL_DB_PATH": str(self.tmp_path / "hitl.db"),
            "HITL_KNOWLEDGE_PATH": str(self.tmp_path / "chroma"),
            "HITL_BOOKING_PATH": str(self.tmp_path / "bookings.json"),
            "HITL_RUNTIME_DIR": str(self.tmp_path),
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(_HITL_SCRIPT), *args], env=self.env, capture_output=True, text=True, timeout=60
        )

    def test_pause_new_process_resume_approve(self) -> None:
        """ADV-08/ADV-11: пауза -> НОВИЙ процес -> resume; повторний resume
        (replay) без дублю — усе через реальний subprocess.run.
        """
        pause_result = self._run("pause")
        self.assertEqual(pause_result.returncode, 0, msg=pause_result.stdout + pause_result.stderr)
        self.assertIn("Interrupt payload", pause_result.stdout)

        resume_result = self._run("resume", "approve")
        self.assertEqual(resume_result.returncode, 0, msg=resume_result.stdout + resume_result.stderr)
        self.assertIn("BOOK-001", resume_result.stdout)
        self.assertIn("без дублю", resume_result.stdout)

    def test_all_in_one_scenarios_via_cli(self) -> None:
        for scenario in ("approve", "reject", "edit", "invalid-edit"):
            with self.subTest(scenario=scenario):
                result = self._run(scenario)
                self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
