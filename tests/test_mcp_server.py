"""Перевірки mcp server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp_server.py"


def _server_params(storage_root: Path) -> StdioServerParameters:
    env = {
        **os.environ,
        "MCP_KNOWLEDGE_PATH": str(storage_root / "chroma"),
        "MCP_BOOKING_PATH": str(storage_root / "bookings.json"),
    }
    return StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)], env=env)


@asynccontextmanager
async def open_session(storage_root: Path) -> AsyncIterator[ClientSession]:
    """Підняти mcp_server.py як окремий subprocess і повернути ініціалізовану сесію."""
    async with stdio_client(_server_params(storage_root)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


# --- list_tools: точний набір і схеми ---------------------------------------


@pytest.mark.asyncio
async def test_list_tools_exact_set_and_schemas(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        result = await session.list_tools()
        names = {tool.name for tool in result.tools}
        assert names == {
            "get_weather",
            "calculate_trip_cost",
            "find_activities",
            "search_knowledge",
            "book_hotel",
        }

        by_name = {tool.name: tool for tool in result.tools}

        weather_schema = by_name["get_weather"].inputSchema
        assert weather_schema["required"] == ["city"]  # days має default і не є required
        assert set(weather_schema["properties"]) == {"city", "days"}

        booking_schema = by_name["book_hotel"].inputSchema
        assert set(booking_schema["required"]) == {"hotel_name", "check_in", "nights", "total_cost"}
        assert booking_schema["properties"]["currency"]["default"] == "EUR"


# --- weather і cost: коректний результат на fixtures ------------------------


@pytest.mark.asyncio
async def test_get_weather_fixture(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        result = await session.call_tool("get_weather", {"city": "Київ", "days": 3})
        assert result.isError is False
        text = result.content[0].text
        assert "18°C" in text
        assert "3 дн." in text


@pytest.mark.asyncio
async def test_calculate_trip_cost_fixture(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        result = await session.call_tool(
            "calculate_trip_cost", {"city": "Львів", "days": 3, "travelers": 2}
        )
        assert result.isError is False
        assert "312 EUR" in result.content[0].text


# --- activities і search_knowledge: змістовна відповідь, document IDs ------


@pytest.mark.asyncio
async def test_find_activities_meaningful_response(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        result = await session.call_tool("find_activities", {"city": "Одеса", "interest": "культура"})
        assert result.isError is False
        text = result.content[0].text
        assert "Одеса" in text
        assert "музей" in text


@pytest.mark.asyncio
async def test_search_knowledge_returns_document_ids(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        result = await session.call_tool("search_knowledge", {"query": "Коли зачинений Лувр?"})
        assert result.isError is False
        text = result.content[0].text
        assert "Лувр" in text
        assert "[travel-" in text  # document ID для атрибуції джерела
        assert len(text.split("\n---\n")) == 3


# --- невалідні city/days/cost/date: помилка без зміни storage ---------------


@pytest.mark.asyncio
async def test_invalid_city_rejected(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        result = await session.call_tool("get_weather", {"city": "Атлантида", "days": 1})
        assert result.isError is True
        assert "не підтримується" in result.content[0].text


@pytest.mark.asyncio
async def test_invalid_days_rejected(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        result = await session.call_tool("get_weather", {"city": "Київ", "days": 99})
        assert result.isError is True
        assert "від 1 до 7" in result.content[0].text


@pytest.mark.asyncio
async def test_invalid_booking_rejected_without_storage_side_effect(tmp_path: Path) -> None:
    booking_path = tmp_path / "bookings.json"
    async with open_session(tmp_path) as session:
        result = await session.call_tool(
            "book_hotel",
            {
                "hotel_name": "A",  # коротше мінімуму
                "check_in": "15.09.2026",  # неправильний формат дати
                "nights": 0,  # поза діапазоном
                "total_cost": -1,  # поза діапазоном
            },
        )
        assert result.isError is True
        text = result.content[0].text
        assert "hotel_name" in text and "check_in" in text and "nights" in text and "total_cost" in text
    assert not booking_path.exists()  # жодного запису не створено


# --- booking: запис створено один раз; повтор operation ID не дублює -------


@pytest.mark.asyncio
async def test_booking_created_once_and_repeat_not_duplicated(tmp_path: Path) -> None:
    booking_path = tmp_path / "bookings.json"
    args = {
        "hotel_name": "Hotel Lumiere",
        "check_in": "2026-09-15",
        "nights": 2,
        "total_cost": 280,
    }
    async with open_session(tmp_path) as session:
        first = await session.call_tool("book_hotel", args)
        assert first.isError is False
        assert "BOOK-001" in first.content[0].text
        stored = json.loads(booking_path.read_text(encoding="utf-8"))
        assert len(stored) == 1
        assert stored[0]["status"] == "confirmed"

        second = await session.call_tool("book_hotel", args)
        assert second.isError is False
        assert "BOOK-001" in second.content[0].text
        assert "вже існує" in second.content[0].text
        stored_after_repeat = json.loads(booking_path.read_text(encoding="utf-8"))
        assert len(stored_after_repeat) == 1  # повтор не задублював запис


# --- list_resources/read_resource: обидва URI, вміст не змінює стан --------


@pytest.mark.asyncio
async def test_list_and_read_resources(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        listed = await session.list_resources()
        uris = {str(resource.uri) for resource in listed.resources}
        assert uris == {"faq://general", "travel://cities"}

        faq = await session.read_resource("faq://general")
        faq_payload = json.loads(faq.contents[0].text)
        assert isinstance(faq_payload, list)
        assert len(faq_payload) >= 1
        assert "question" in faq_payload[0] and "answer" in faq_payload[0]

        cities = await session.read_resource("travel://cities")
        cities_payload = json.loads(cities.contents[0].text)
        assert "Київ" in cities_payload
        assert "daily_eur" in cities_payload["Київ"]

        # Повторне read того самого URI не має side effect: вміст ідентичний.
        faq_again = await session.read_resource("faq://general")
        assert faq_again.contents[0].text == faq.contents[0].text


# --- list_prompts/get_prompt: шаблон і параметри ----------------------------


@pytest.mark.asyncio
async def test_list_and_get_prompt_template_and_parameters(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        listed = await session.list_prompts()
        assert {prompt.name for prompt in listed.prompts} == {"travel_reply"}
        (prompt_def,) = listed.prompts
        assert {arg.name for arg in prompt_def.arguments} == {"traveler_name", "issue_summary", "tone"}

        result = await session.get_prompt(
            "travel_reply",
            {
                "traveler_name": "Олена",
                "issue_summary": "Загубився багаж в аеропорту",
                "tone": "формальний",
            },
        )
        text = result.messages[0].content.text
        assert "Олена" in text
        assert "Загубився багаж в аеропорту" in text
        # Це шаблон, а не готова вигадана відповідь: він вказує агенту додати факти.
        assert "get_weather" in text


@pytest.mark.asyncio
async def test_get_prompt_invalid_tone_rejected(tmp_path: Path) -> None:
    async with open_session(tmp_path) as session:
        with pytest.raises(Exception):
            await session.get_prompt(
                "travel_reply",
                {
                    "traveler_name": "Олена",
                    "issue_summary": "Питання про маршрут",
                    "tone": "агресивний",
                },
            )


# --- stdio smoke integration: окремий subprocess, без зависання -------------


@pytest.mark.asyncio
async def test_stdio_subprocess_completes_without_hanging(tmp_path: Path) -> None:
    """Підтверджує, що stdio transport коректно стартує й зупиняється в межах
    тайм-ауту, а не зависає (окремий явний timeout-guard понад звичайний тест).
    """

    async def run() -> None:
        async with open_session(tmp_path) as session:
            result = await session.list_tools()
            assert len(result.tools) == 5

    await asyncio.wait_for(run(), timeout=20)
