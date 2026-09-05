"""Тести data/ (S16): синтетичні fixtures не розходяться з кодом, що їх
використовує чи документує."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from knowledge import KNOWLEDGE_DOCUMENTS

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class KnowledgeSeedSnapshotTests(unittest.TestCase):
    """data/knowledge_seed.json -- довідковий знімок KNOWLEDGE_DOCUMENTS
    (S16: data/ обов'язковий у структурі здачі). Джерело істини лишається
    knowledge.py; цей тест ловить розсинхронізацію, якщо документи
    змінили в одному місці й забули в іншому."""

    def test_seed_matches_knowledge_documents(self) -> None:
        payload = json.loads((_DATA_DIR / "knowledge_seed.json").read_text(encoding="utf-8"))
        docs = payload["documents"]
        self.assertEqual(len(docs), len(KNOWLEDGE_DOCUMENTS))
        for index, (doc, text) in enumerate(zip(docs, KNOWLEDGE_DOCUMENTS), start=1):
            with self.subTest(index=index):
                self.assertEqual(doc["id"], f"travel-{index:02d}")
                self.assertEqual(doc["text"], text)


class RedTeamFixturesTests(unittest.TestCase):
    def test_rt02_and_rt06_fixtures_load_and_have_expected_shape(self) -> None:
        payload = json.loads((_DATA_DIR / "red_team_fixtures.json").read_text(encoding="utf-8"))
        rt02 = payload["rt02_pii_document"]
        rt06 = payload["rt06_injection_document"]

        self.assertIn("{email}", rt02["text_template"])
        self.assertIn("{phone}", rt02["text_template"])
        self.assertIn("@", rt02["synthetic_email"])

        self.assertIn("Ignore all previous instructions", rt06["text"])
        self.assertIn("book_hotel", rt06["text"])


if __name__ == "__main__":
    unittest.main()
