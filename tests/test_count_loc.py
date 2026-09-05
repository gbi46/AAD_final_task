"""Тест count_loc.py (S15): фіксоване правило LOC (не порожні, не чисто
коментарні рядки) застосовується детерміновано."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from count_loc import count_loc


class CountLocTests(unittest.TestCase):
    def test_blank_and_comment_only_lines_excluded(self) -> None:
        content = (
            "# module docstring header comment\n"
            "\n"
            "import os\n"
            "  # indented full-line comment\n"
            "\n"
            "def f():\n"
            "    return 1  # trailing comment counts (not a comment-only line)\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(content)
            path = Path(fh.name)
        try:
            # Рядки, що рахуються: "import os", "def f():",
            # "    return 1  # trailing comment counts (...)" -> 3.
            self.assertEqual(count_loc(path), 3)
        finally:
            path.unlink()

    def test_empty_file_is_zero(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            path = Path(fh.name)
        try:
            self.assertEqual(count_loc(path), 0)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
