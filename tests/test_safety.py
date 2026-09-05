"""Тести захисних механізмів.

Перенесено без змін з `Kravchuk_AAD_1/tests/test_safety.py`: ліміт кроків,
тайм-аут, детектор повторюваних викликів (repeat limit).
"""

from __future__ import annotations

import time
import unittest

from safety import ExecutionGuard, SafetyStop


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


if __name__ == "__main__":
    unittest.main()
