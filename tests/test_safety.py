"""Перевірки safety."""

from __future__ import annotations

import time
import unittest

from safety import ExecutionGuard, SafetyStop, call_with_timeout


class SafetyTests(unittest.TestCase):
    def test_max_steps(self) -> None:
        guard = ExecutionGuard(max_steps=2)
        with self.assertRaisesRegex(SafetyStop, "ліміт кроків"):
            guard.check_limits(2)

    def test_timeout(self) -> None:
        guard = ExecutionGuard(timeout_seconds=0.001)
        time.sleep(0.002)
        with self.assertRaisesRegex(SafetyStop, "тайм-аут"):
            guard.check_limits(0)

    def test_repeated_calls(self) -> None:
        guard = ExecutionGuard(max_repeats=3)
        guard.register_call("get_weather", {"city": "Київ"})
        guard.register_call("get_weather", {"city": "Київ"})
        with self.assertRaisesRegex(SafetyStop, "цикл"):
            guard.register_call("get_weather", {"city": "Київ"})

    def test_call_with_timeout_returns_result_when_fast_enough(self) -> None:
        self.assertEqual(call_with_timeout(lambda: "ok", 5.0), "ok")

    def test_call_with_timeout_stops_waiting_on_hanging_call(self) -> None:
        def hangs() -> str:
            time.sleep(2.0)
            return "too-late"

        started = time.monotonic()
        with self.assertRaisesRegex(SafetyStop, "тайм-аут"):
            call_with_timeout(hangs, 0.2)
        elapsed = time.monotonic() - started
        # Абонент повертає керування біля межі тайм-ауту, а не чекає всі 2 с.
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
