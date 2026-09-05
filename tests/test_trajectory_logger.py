"""Тести розширеного TrajectoryLogger: agent_name, санітизація, повнота кроків.

Новий тестовий файл (у ДЗ1 подібного модульного тесту не було — trajectory_logger
перевірявся лише опосередковано через agent.py). Перевіряє саме зміни S02.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trajectory_logger import TrajectoryLogger, save_trajectories


class TrajectoryLoggerTests(unittest.TestCase):
    def test_agent_name_recorded_with_existing_fields(self) -> None:
        logger = TrajectoryLogger("run-1")
        logger.log(
            node="agent",
            input_data="привіт",
            output_data="відповідь",
            tool_calls=[{"name": "get_weather", "args": {"city": "Київ"}}],
            duration_ms=42,
            agent_name="tech",
        )
        step = logger.as_dict()["trajectory"][0]
        self.assertEqual(step["agent_name"], "tech")
        self.assertEqual(step["node_name"], "agent")
        self.assertEqual(step["duration_ms"], 42)
        self.assertEqual(step["tool_calls"][0]["name"], "get_weather")
        self.assertIn("timestamp", step)

    def test_agent_name_defaults_when_not_provided(self) -> None:
        logger = TrajectoryLogger("run-2")
        logger.log(node="agent", input_data="x", output_data="y")
        self.assertEqual(logger.as_dict()["trajectory"][0]["agent_name"], "supervisor")

    def test_sensitive_values_are_redacted(self) -> None:
        logger = TrajectoryLogger("run-3")
        leaked_key = "AIza" + "A" * 35
        logger.log(
            node="agent",
            input_data=f"мій ключ {leaked_key}",
            output_data="ok",
            tool_calls=[{"name": "call", "args": {"token": f"Bearer {leaked_key}xyz1234567"}}],
        )
        step = logger.as_dict()["trajectory"][0]
        self.assertNotIn(leaked_key, step["input"])
        self.assertIn("REDACTED", step["input"])
        self.assertNotIn(leaked_key, json.dumps(step["tool_calls"]))

    def test_long_output_is_not_collapsed_into_short_summary(self) -> None:
        long_text = "спостереження " * 100  # довший за старий поріг 500 символів
        logger = TrajectoryLogger("run-4")
        logger.log(node="tools", input_data="q", output_data=long_text)
        stored = logger.as_dict()["trajectory"][0]["output"]
        self.assertGreater(len(stored), 500)
        self.assertIn(long_text[:500], stored)

    def test_save_trajectories_writes_json_file(self) -> None:
        logger = TrajectoryLogger("run-5")
        logger.log(node="agent", input_data="q", output_data="a", agent_name="billing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            save_trajectories([logger.as_dict()], path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["runs"][0]["run_id"], "run-5")
            self.assertEqual(payload["runs"][0]["trajectory"][0]["agent_name"], "billing")


if __name__ == "__main__":
    unittest.main()
