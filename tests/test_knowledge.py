"""Перевірки knowledge."""

from __future__ import annotations

import tempfile
import unittest

from knowledge import KnowledgeBase


class KnowledgeTests(unittest.TestCase):
    def test_collection_contains_ten_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = KnowledgeBase(directory)
            self.assertEqual(base.count, 10)

    def test_search_returns_three_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = KnowledgeBase(directory).search("правила бронювання готелю")
            self.assertEqual(len(documents), 3)
            self.assertTrue(any("бронювання" in document.lower() for document in documents))


if __name__ == "__main__":
    unittest.main()
