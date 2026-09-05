"""Перевірки persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "persistence_demo.py"


def _env_for(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PERSISTENCE_DB_PATH": str(tmp_path / "agent_state.db"),
        "PERSISTENCE_KNOWLEDGE_PATH": str(tmp_path / "chroma_db"),
        "PERSISTENCE_TRAJECTORY_PATH": str(tmp_path / "trajectory.json"),
        "PERSISTENCE_RUNTIME_DIR": str(tmp_path / "runtime"),
    }


def _run_phase(phase: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), phase],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class PersistenceDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp.name)
        self.env = _env_for(self.tmp_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_pause_then_resume_across_separate_processes(self) -> None:
        """MAS-15/16/17: файловий SqliteSaver, pause після одного кроку,
        новий процес відновлює той самий checkpoint і доводить план до
        кінця без повторного виконання вже зробленого кроку.
        """
        pause_result = _run_phase("pause", self.env)
        self.assertEqual(pause_result.returncode, 0, msg=pause_result.stdout + pause_result.stderr)
        self.assertIn("завершено", pause_result.stdout)

        db_path = self.tmp_path / "agent_state.db"
        self.assertTrue(db_path.exists(), "agent_state.db має існувати після фази pause")

        resume_result = _run_phase("resume", self.env)
        self.assertEqual(resume_result.returncode, 0, msg=resume_result.stdout + resume_result.stderr)
        self.assertIn("current_step: 0 -> 2", resume_result.stdout)
        self.assertIn("billing довиконано до кінця", resume_result.stdout)

    def test_resume_without_prior_pause_fails_clearly(self) -> None:
        """Без попередньої фази pause (агент_state.db відсутній) resume не
        повинен мовчки створювати щось нове — явна помилка з підказкою.
        """
        result = _run_phase("resume", self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Спочатку запустіть", result.stdout + result.stderr)

    def test_agent_state_db_is_a_valid_self_contained_sqlite_file(self) -> None:
        """Контрольний артефакт: цілісний.db без потреби у -wal/-shm."""
        _run_phase("pause", self.env)
        _run_phase("resume", self.env)

        db_path = self.tmp_path / "agent_state.db"
        wal_path = db_path.with_name(db_path.name + "-wal")
        shm_path = db_path.with_name(db_path.name + "-shm")
        self.assertFalse(wal_path.exists(), "WAL sidecar не мав лишитись після checkpoint")
        self.assertFalse(shm_path.exists())

        connection = sqlite3.connect(str(db_path))
        try:
            self.assertEqual(connection.execute("PRAGMA integrity_check;").fetchone(), ("ok",))
            self.assertEqual(connection.execute("PRAGMA journal_mode;").fetchone(), ("delete",))
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table';")}
            self.assertIn("checkpoints", tables)
        finally:
            connection.close()

    def test_trajectory_json_has_both_phases_with_agent_name(self) -> None:
        """Контрольний артефакт: trajectory.json об'єднує обидві фази,
        agent_name присутній у кожному кроці кожної фази.
        """
        _run_phase("pause", self.env)
        _run_phase("resume", self.env)

        trajectory_path = self.tmp_path / "trajectory.json"
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["runs"]), 2)
        for run in payload["runs"]:
            for step in run["trajectory"]:
                self.assertTrue(step.get("agent_name"))

    def test_different_thread_id_stays_isolated_across_processes(self) -> None:
        """MAS-17: інший thread_id не бачить стан першого — обидві фази
        самі перевіряють це через assert (subprocess впав би з ненульовим
        exit code, якби isolation порушилась); тут підтверджуємо, що
        відповідні перевірки в скрипті справді виконались (є в stdout).
        """
        pause_result = _run_phase("pause", self.env)
        self.assertEqual(pause_result.returncode, 0, msg=pause_result.stdout + pause_result.stderr)
        self.assertIn("ізоляція сесій", pause_result.stdout)

        resume_result = _run_phase("resume", self.env)
        self.assertEqual(resume_result.returncode, 0, msg=resume_result.stdout + resume_result.stderr)
        self.assertIn("не бачить стану першого", resume_result.stdout)


if __name__ == "__main__":
    unittest.main()
