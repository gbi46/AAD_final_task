"""Перевірки tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from tools_legacy import (
    ActivityInput,
    BookingInput,
    KnowledgeInput,
    TripCostInput,
    WeatherInput,
    book_hotel,
    calculate_trip_cost,
    configure_storage,
    find_activities,
    get_weather,
    search_knowledge,
)


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        configure_storage(root / "chroma", root / "bookings.json")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_weather(self) -> None:
        result = get_weather.invoke({"city": "Київ", "days": 3})
        self.assertIn("18°C", result)
        self.assertIn("3 дн.", result)
        with self.assertRaisesRegex(ValidationError, "від 1 до 7"):
            WeatherInput(city="Київ", days=8)

    def test_trip_cost(self) -> None:
        result = calculate_trip_cost.invoke({"city": "Львів", "days": 3, "travelers": 2})
        self.assertIn("312 EUR", result)
        with self.assertRaisesRegex(ValidationError, "від 1 до 10"):
            TripCostInput(city="Львів", days=3, travelers=0)

    def test_activities(self) -> None:
        result = find_activities.invoke({"city": "Одеса", "interest": "культура"})
        self.assertIn("головний музей", result)
        with self.assertRaises(ValidationError):
            ActivityInput(city="Невідоме", interest="природа")

    def test_agentic_rag_tool(self) -> None:
        result = search_knowledge.invoke({"query": "Коли зачинений Лувр?"})
        self.assertIn("Лувр", result)
        self.assertIn("[travel-", result)  # ID документа для атрибуції джерела
        self.assertEqual(len(result.split("\n---\n")), 3)
        with self.assertRaisesRegex(ValidationError, "від 3 до 300"):
            KnowledgeInput(query="ок")

    def test_booking_side_effect_and_idempotency(self) -> None:
        args = {
            "hotel_name": "Hotel Lumiere",
            "check_in": "2026-09-15",
            "nights": 2,
            "total_cost": 280,
        }
        first = book_hotel.invoke(args)
        self.assertIn("BOOK-001", first)
        stored = json.loads((Path(self.temp.name) / "bookings.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["status"], "confirmed")

        # Повторний виклик з тими самими параметрами не створює друге бронювання.
        second = book_hotel.invoke(args)
        self.assertIn("BOOK-001", second)
        self.assertIn("вже існує", second)
        stored_after_repeat = json.loads(
            (Path(self.temp.name) / "bookings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(stored_after_repeat), 1)

        # Інші параметри — нове бронювання з новим ідентифікатором.
        third = book_hotel.invoke({**args, "nights": 3})
        self.assertIn("BOOK-002", third)
        stored_after_new = json.loads(
            (Path(self.temp.name) / "bookings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(stored_after_new), 2)

    def test_invalid_inputs_are_rejected(self) -> None:
        invalid = (
            lambda: WeatherInput(city="Прага", days=1),
            lambda: WeatherInput(city="Київ", days=8),
            lambda: TripCostInput(city="Львів", days=0, travelers=1),
            lambda: KnowledgeInput(query="x"),
            lambda: BookingInput(hotel_name="A", check_in="15.09.2026", nights=0, total_cost=-1),
        )
        for construct in invalid:
            with self.subTest(construct=construct), self.assertRaises(ValidationError):
                construct()


if __name__ == "__main__":
    unittest.main()
