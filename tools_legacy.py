"""Валідовані туристичні інструменти — спільне ядро для MAS-агентів і MCP.

Об'єднано й адаптовано з двох попередніх домашніх завдань (детальніше — у
таблиці походження компонентів у README):

- `Kravchuk_AAD_1/domain_tools.py` (ДЗ1, Завд. 1): `get_weather`,
  `estimate_budget`, `find_activities`, каталог міст CITY_DATA.
- `Kravchuk_AAD_2/domain_tools.py` (ДЗ2, Завд. 1/3): `calculate_trip_cost`
  (те саме обчислення, що й `estimate_budget`), `search_knowledge` (Agentic
  RAG над `knowledge.KnowledgeBase`), `book_hotel` (ризикова дія для HITL).

Каталог міст об'єднано без конфліктів — Київ/Львів/Одеса/Берлін мають
однакові значення в обох джерелах; додано Варшаву (ДЗ1) і Париж (ДЗ2), бо
документи `knowledge.py` посилаються саме на Париж і Берлін.

Рівно п'ять публічних MCP-tools (буде підключено в MCP-сервері, S03):
`get_weather`, `calculate_trip_cost`, `find_activities`, `search_knowledge`,
`book_hotel`. `convert_currency` з ДЗ1 навмисно не перенесено — він не
входить у цей список і дублював би поверхню MCP-сервера без потреби.

Кожен tool має тонку LangChain-обгортку (`@tool(args_schema=...)`) над
"core"-функцією з тією самою бізнес-логікою. MCP-сервер (S03) імпортує й
викликає ці core-функції та Pydantic-схеми напряму, не залежачи від
LangChain `BaseTool` — так одна реалізація обслуговує і агентів LangGraph, і
MCP-клієнтів.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from knowledge import KnowledgeBase


CITY_DATA = {
    "Київ": {"daily_eur": 58, "temperature": 18, "condition": "мінлива хмарність"},
    "Львів": {"daily_eur": 52, "temperature": 15, "condition": "дощ"},
    "Одеса": {"daily_eur": 64, "temperature": 23, "condition": "сонячно"},
    "Варшава": {"daily_eur": 78, "temperature": 17, "condition": "хмарно"},
    "Берлін": {"daily_eur": 105, "temperature": 16, "condition": "легкий дощ"},
    "Париж": {"daily_eur": 145, "temperature": 19, "condition": "хмарно"},
}

_knowledge_base: KnowledgeBase | None = None
_knowledge_path = Path("chroma_db")
_booking_path = Path("bookings.json")


def configure_storage(knowledge_path: str | Path, booking_path: str | Path) -> None:
    """Налаштувати runtime-шляхи сховищ без глобального стану під час імпорту.

    Викликається на старті застосунку (і в кожному тесті окремо, з
    tempfile-директорією) — інакше всі виклики writе/read йшли б у спільні
    `chroma_db/`/`bookings.json` у поточній директорії.
    """
    global _knowledge_base, _knowledge_path, _booking_path
    _knowledge_path = Path(knowledge_path)
    _booking_path = Path(booking_path)
    _knowledge_base = None


def get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(_knowledge_path)
    return _knowledge_base


# --- Pydantic-схеми --------------------------------------------------------


class CityInput(BaseModel):
    """Спільна схема для підтримуваного міста."""

    model_config = ConfigDict(str_strip_whitespace=True)
    city: str = Field(description="Назва підтримуваного міста українською мовою")

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        if value not in CITY_DATA:
            raise ValueError(f"Місто не підтримується. Доступні: {', '.join(CITY_DATA)}")
        return value


class WeatherInput(CityInput):
    """Параметри прогнозу погоди."""

    days: int = Field(default=1, description="Кількість днів прогнозу від 1 до 7")

    @field_validator("days")
    @classmethod
    def validate_days(cls, value: int) -> int:
        if not 1 <= value <= 7:
            raise ValueError("Кількість днів має бути від 1 до 7")
        return value


class TripCostInput(CityInput):
    """Параметри розрахунку бюджету подорожі."""

    days: int = Field(description="Тривалість подорожі від 1 до 30 днів")
    travelers: int = Field(default=1, description="Кількість мандрівників від 1 до 10")

    @field_validator("days")
    @classmethod
    def validate_days(cls, value: int) -> int:
        if not 1 <= value <= 30:
            raise ValueError("Тривалість має бути від 1 до 30 днів")
        return value

    @field_validator("travelers")
    @classmethod
    def validate_travelers(cls, value: int) -> int:
        if not 1 <= value <= 10:
            raise ValueError("Кількість мандрівників має бути від 1 до 10")
        return value


class ActivityInput(CityInput):
    """Параметри пошуку активностей."""

    interest: Literal["культура", "їжа", "природа"] = Field(description="Категорія інтересу")

    @field_validator("interest")
    @classmethod
    def normalize_interest(cls, value: str) -> str:
        return value.lower()


class KnowledgeInput(BaseModel):
    """Параметри семантичного пошуку туристичних правил і FAQ."""

    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(description="Довідкове туристичне питання довжиною 3–300 символів")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not 3 <= len(value) <= 300:
            raise ValueError("Пошуковий запит має містити від 3 до 300 символів")
        return value


class BookingInput(BaseModel):
    """Параметри ризикового бронювання готелю (лише локальна демонстрація)."""

    model_config = ConfigDict(str_strip_whitespace=True)
    hotel_name: str = Field(description="Назва готелю довжиною 2–100 символів")
    check_in: str = Field(description="Дата заїзду у форматі YYYY-MM-DD")
    nights: int = Field(description="Кількість ночей від 1 до 30")
    total_cost: float = Field(description="Фінальна вартість у EUR від 1 до 20 000")
    currency: Literal["EUR"] = Field(default="EUR", description="Валюта бронювання — EUR")

    @field_validator("hotel_name")
    @classmethod
    def validate_hotel(cls, value: str) -> str:
        if not 2 <= len(value) <= 100:
            raise ValueError("Назва готелю має містити від 2 до 100 символів")
        return value

    @field_validator("check_in")
    @classmethod
    def validate_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Дата заїзду має формат YYYY-MM-DD") from exc
        return value

    @field_validator("nights")
    @classmethod
    def validate_nights(cls, value: int) -> int:
        if not 1 <= value <= 30:
            raise ValueError("Кількість ночей має бути від 1 до 30")
        return value

    @field_validator("total_cost")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        if not 1 <= value <= 20_000:
            raise ValueError("Вартість має бути від 1 до 20 000 EUR")
        return round(value, 2)


# --- Core-функції (спільне ядро для LangChain і MCP) ------------------------


def get_weather_core(city: str, days: int = 1) -> str:
    data = CITY_DATA[city]
    return f"{city}: {data['temperature']}°C, {data['condition']}; прогноз на {days} дн."


def calculate_trip_cost_core(city: str, days: int, travelers: int = 1) -> str:
    total = CITY_DATA[city]["daily_eur"] * days * travelers
    return f"Орієнтовний бюджет для {city}: {total} EUR ({days} дн., {travelers} ос.)."


def find_activities_core(city: str, interest: str) -> str:
    catalog = {
        "культура": "історичний центр; головний музей",
        "їжа": "локальний ринок; гастрономічна екскурсія",
        "природа": "міський парк; прогулянка біля води",
    }
    return f"{city}, {interest}: {catalog[interest]}."


def search_knowledge_core(query: str, limit: int = 3) -> str:
    documents = get_knowledge_base().search(query, limit=limit)
    return "\n---\n".join(documents)


def _idempotency_key(hotel_name: str, check_in: str, nights: int, total_cost: float, currency: str) -> str:
    """Детермінований ідентифікатор операції бронювання.

    Той самий набір параметрів завжди дає той самий ключ — повторний виклик
    (наприклад, через retry супервізора після мережевого збою) не створює
    другого бронювання, а повертає вже наявний запис.
    """
    payload = json.dumps(
        {
            "hotel_name": hotel_name,
            "check_in": check_in,
            "nights": nights,
            "total_cost": total_cost,
            "currency": currency,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_bookings() -> list[dict]:
    if _booking_path.exists():
        return json.loads(_booking_path.read_text(encoding="utf-8"))
    return []


def _save_bookings(bookings: list[dict]) -> None:
    _booking_path.parent.mkdir(parents=True, exist_ok=True)
    _booking_path.write_text(
        json.dumps(bookings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def book_hotel_core(
    hotel_name: str,
    check_in: str,
    nights: int,
    total_cost: float,
    currency: str = "EUR",
) -> str:
    validated = BookingInput(
        hotel_name=hotel_name,
        check_in=check_in,
        nights=nights,
        total_cost=total_cost,
        currency=currency,
    )
    key = _idempotency_key(
        validated.hotel_name, validated.check_in, validated.nights, validated.total_cost, validated.currency
    )
    bookings = _load_bookings()
    existing = next((item for item in bookings if item.get("idempotency_key") == key), None)
    if existing is not None:
        return (
            f"Бронювання {existing['booking_id']} вже існує (ідемпотентний повтор): "
            f"{existing['hotel_name']}, {existing['check_in']}, {existing['nights']} ноч., "
            f"{existing['total_cost']:.2f} {existing['currency']}."
        )
    booking = validated.model_dump()
    booking["idempotency_key"] = key
    booking["booking_id"] = f"BOOK-{len(bookings) + 1:03d}"
    booking["status"] = "confirmed"
    bookings.append(booking)
    _save_bookings(bookings)
    return (
        f"Бронювання {booking['booking_id']} підтверджено: {booking['hotel_name']}, "
        f"{booking['check_in']}, {booking['nights']} ноч., {booking['total_cost']:.2f} {booking['currency']}."
    )


# --- LangChain-обгортки (публічний інтерфейс для агентів) -------------------


@tool(args_schema=WeatherInput)
def get_weather(city: str, days: int = 1) -> str:
    """Отримати прогноз погоди для планування маршруту чи одягу.

    Викликайте перед порадами про одяг, дощові дні або маршрут. Не
    використовуйте для правил, бюджету чи бронювання.
    """
    return get_weather_core(city, days)


@tool(args_schema=TripCostInput)
def calculate_trip_cost(city: str, days: int, travelers: int = 1) -> str:
    """Розрахувати орієнтовний бюджет подорожі у EUR.

    Використовуйте для арифметичної оцінки проживання, харчування й
    транспорту за заданої тривалості та кількості мандрівників.
    """
    return calculate_trip_cost_core(city, days, travelers)


@tool(args_schema=ActivityInput)
def find_activities(city: str, interest: str) -> str:
    """Знайти дві доречні активності у місті за інтересом: культура, їжа або природа."""
    return find_activities_core(city, interest)


@tool(args_schema=KnowledgeInput)
def search_knowledge(query: str) -> str:
    """Знайти туристичні правила, довідкові факти чи рекомендації у базі знань.

    Використовуйте для візових правил, транспорту, безпеки, сезонності,
    музеїв і рекомендацій. Не використовуйте для погоди, обчислень чи
    бронювання. Повертає до трьох релевантних документів.
    """
    return search_knowledge_core(query, limit=3)


@tool(args_schema=BookingInput)
def book_hotel(
    hotel_name: str,
    check_in: str,
    nights: int,
    total_cost: float,
    currency: str = "EUR",
) -> str:
    """Забронювати готель після явного HITL-підтвердження.

    Це ризикова дія з фінансовими наслідками (лише локальна демонстрація,
    без реальної інтеграції з провайдером бронювань). Виклик ідемпотентний:
    повторний виклик з тими самими параметрами не створює дублікат.
    Executor зобов'язаний викликати interrupt перед виконанням цього
    інструмента (буде реалізовано в S09 — HITL).
    """
    return book_hotel_core(hotel_name, check_in, nights, total_cost, currency)


TOOLS = [get_weather, calculate_trip_cost, find_activities, search_knowledge, book_hotel]
TOOLS_BY_NAME = {item.name: item for item in TOOLS}
RISKY_TOOLS = {"book_hotel"}
