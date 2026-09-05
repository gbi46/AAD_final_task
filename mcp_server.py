"""MCP-сервер туристичної підтримки: пʼять tools, два resources і prompt.

Вхідні дані повторно валідуються Pydantic-схемами зі спільного доменного
модуля. Транспорт — stdio; журнали спрямовуються у stderr.
MCP_KNOWLEDGE_PATH і MCP_BOOKING_PATH задають локальні сховища."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError

import tools_legacy as domain
from tools_legacy import CITY_DATA, ActivityInput, BookingInput, KnowledgeInput, TripCostInput, WeatherInput


# --- Логування: stdout належить протоколу, логи — у stderr -----------------

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mcp_server")


# --- Сховища: шлях від розташування файлу, перевизначається через env -------

_project_root = Path(__file__).resolve().parent
KNOWLEDGE_PATH = Path(os.environ.get("MCP_KNOWLEDGE_PATH", str(_project_root / "chroma_db")))
BOOKING_PATH = Path(os.environ.get("MCP_BOOKING_PATH", str(_project_root / "bookings.json")))
domain.configure_storage(KNOWLEDGE_PATH, BOOKING_PATH)


def _validation_error_message(exc: ValidationError) -> str:
    """Компактне повідомлення для клієнта без внутрішнього трейсбеку pydantic."""
    parts = [f"{'/'.join(str(item) for item in error['loc'])}: {error['msg']}" for error in exc.errors()]
    return "; ".join(parts)


mcp = FastMCP(
    name="travel-support",
    instructions=(
        "MCP-сервер туристичної координації: погода, орієнтовний бюджет, "
        "активності, довідкові правила (RAG) і демонстраційне бронювання "
        "готелю. Усі tools повторно валідують аргументи на сервері."
    ),
)


# --- Tools -------------------------------------------------------------------


@mcp.tool()
def get_weather(city: str, days: int = 1) -> str:
    """Отримати прогноз погоди для планування маршруту чи одягу.

    Коли використовувати: перед порадами про одяг, дощові дні чи маршрут.
    Не використовувати для бюджету, активностей чи довідкових правил.

    Args:
        city: Назва підтримуваного міста українською (Київ, Львів, Одеса,
            Варшава, Берлін, Париж). Повний перелік — resource `travel://cities`.
        days: Кількість днів прогнозу від 1 до 7 (типово 1).

    Returns:
        Короткий текстовий прогноз: температура, умови, горизонт прогнозу.

    Помилки:
        Непідтримуване місто чи `days` поза діапазоном 1–7 підіймають
        ValueError з описом проблеми (сервер сам ще раз валідує Pydantic-схемою,
        навіть якщо клієнт уже перевірив JSON Schema).
    """
    try:
        validated = WeatherInput(city=city, days=days)
    except ValidationError as exc:
        raise ValueError(_validation_error_message(exc)) from exc
    return domain.get_weather_core(validated.city, validated.days)


@mcp.tool()
def calculate_trip_cost(city: str, days: int, travelers: int = 1) -> str:
    """Розрахувати орієнтовний бюджет подорожі у EUR.

    Коли використовувати: арифметична оцінка проживання, харчування й
    транспорту за заданої тривалості та кількості мандрівників.

    Args:
        city: Назва підтримуваного міста українською.
        days: Тривалість подорожі від 1 до 30 днів.
        travelers: Кількість мандрівників від 1 до 10 (типово 1).

    Returns:
        Текст з орієнтовною сумою в EUR, містом, кількістю днів та осіб.

    Помилки:
        Невідоме місто чи параметри поза допустимим діапазоном підіймають
        ValueError.
    """
    try:
        validated = TripCostInput(city=city, days=days, travelers=travelers)
    except ValidationError as exc:
        raise ValueError(_validation_error_message(exc)) from exc
    return domain.calculate_trip_cost_core(validated.city, validated.days, validated.travelers)


@mcp.tool()
def find_activities(city: str, interest: str) -> str:
    """Знайти дві доречні активності у місті за категорією інтересу.

    Args:
        city: Назва підтримуваного міста українською.
        interest: Категорія: "культура", "їжа" або "природа".

    Returns:
        Текст із двома активностями, доречними для міста й категорії.

    Помилки:
        Невідоме місто або категорія поза переліком підіймають ValueError.
    """
    try:
        validated = ActivityInput(city=city, interest=interest)
    except ValidationError as exc:
        raise ValueError(_validation_error_message(exc)) from exc
    return domain.find_activities_core(validated.city, validated.interest)


@mcp.tool()
def search_knowledge(query: str) -> str:
    """Знайти туристичні правила, довідкові факти чи рекомендації (Agentic RAG).

    Коли використовувати: візові правила, транспорт, безпека, сезонність,
    музеї, поради щодо бронювання. Не використовувати для погоди, обчислень
    чи бронювання — для цього є окремі tools.

    Args:
        query: Довідкове туристичне питання довжиною 3–300 символів.

    Returns:
        До трьох релевантних документів із persistent ChromaDB у форматі
        `[travel-XX] текст`, розділених рядком "---" — ID дозволяє
        атрибутувати відповідь до конкретного джерела.

    Помилки:
        Запит поза діапазоном довжини підіймає ValueError. Недоступне чи
        пошкоджене сховище ChromaDB підіймає RuntimeError — це не
        перетворюється на порожній "успішний" результат.
    """
    try:
        validated = KnowledgeInput(query=query)
    except ValidationError as exc:
        raise ValueError(_validation_error_message(exc)) from exc
    try:
        return domain.search_knowledge_core(validated.query, limit=3)
    except ValueError:
        raise
    except Exception as exc:  # сховище (ChromaDB) недоступне чи пошкоджене
        raise RuntimeError(f"Помилка бази знань: {exc}") from exc


@mcp.tool()
def book_hotel(
    hotel_name: str,
    check_in: str,
    nights: int,
    total_cost: float,
    currency: str = "EUR",
) -> str:
    """Забронювати готель — ризикова дія з фінансовими наслідками (демо).

    Це лише локальна демонстраційна дія (без реального провайдера
    бронювань). Виклик ідемпотентний: повторний виклик з тими самими
    аргументами не створює дублікат, а повертає наявний запис.

    Увага:
        MAS виконує цей tool лише після HITL-підтвердження.
        Прямий MCP-клієнт відповідає за власну політику доступу.

    Args:
        hotel_name: Назва готелю, 2–100 символів.
        check_in: Дата заїзду у форматі YYYY-MM-DD.
        nights: Кількість ночей від 1 до 30.
        total_cost: Фінальна вартість у EUR від 1 до 20 000.
        currency: Валюта бронювання — лише "EUR".

    Returns:
        Підтвердження з ідентифікатором бронювання (BOOK-NNN) або
        повідомлення, що ідентичне бронювання вже існує.

    Помилки:
        Невалідні параметри (коротка назва, неправильний формат дати,
        вартість поза діапазоном тощо) підіймають ValueError; сховище
        (bookings.json) не змінюється.
    """
    try:
        validated = BookingInput(
            hotel_name=hotel_name,
            check_in=check_in,
            nights=nights,
            total_cost=total_cost,
            currency=currency,
        )
    except ValidationError as exc:
        raise ValueError(_validation_error_message(exc)) from exc
    return domain.book_hotel_core(
        validated.hotel_name, validated.check_in, validated.nights, validated.total_cost, validated.currency
    )


# --- Resources (read-only, без side effects) --------------------------------

FAQ_GENERAL = [
    {
        "question": "Скільки часу можна перебувати в Шенгені без візи?",
        "answer": (
            "До 90 днів протягом 180-денного періоду для громадян України з "
            "біометричним паспортом; без права на працевлаштування."
        ),
    },
    {
        "question": "Що перевірити перед бронюванням готелю?",
        "answer": (
            "Політику скасування, фінальну суму з податками, час заселення "
            "та рейтинг об'єкта."
        ),
    },
    {
        "question": "Яке базове туристичне страхування потрібне?",
        "answer": (
            "Покриття невідкладної медичної допомоги, госпіталізації та "
            "репатріації; активний відпочинок і хронічні захворювання — окремо."
        ),
    },
    {
        "question": "Як зменшити ризик кишенькових крадіжок?",
        "answer": (
            "Тримати документи й кошти окремо, у людних місцях носити рюкзак "
            "спереду, зберігати копії документів у захищеному цифровому сховищі."
        ),
    },
    {
        "question": "Де шукати детальні правила для конкретного міста?",
        "answer": (
            "Викликати tool search_knowledge — він шукає в повній базі знань "
            "і повертає до трьох релевантних документів."
        ),
    },
]


@mcp.resource(
    "faq://general",
    name="general_faq",
    description="Загальні туристичні FAQ (статичний read-only довідник, без RAG-пошуку)",
    mime_type="application/json",
)
def general_faq() -> str:
    """Повернути статичний перелік загальних туристичних питань і відповідей.

    На відміну від tool `search_knowledge`, цей resource не виконує
    семантичний пошук — це фіксований довідковий контент.
    """
    return json.dumps(FAQ_GENERAL, ensure_ascii=False, indent=2)


@mcp.resource(
    "travel://cities",
    name="supported_cities",
    description="Підтримувані міста та демонстраційні дані (температура, умови, вартість/добу в EUR)",
    mime_type="application/json",
)
def supported_cities() -> str:
    """Повернути каталог підтримуваних міст із демонстраційними даними CITY_DATA."""
    return json.dumps(CITY_DATA, ensure_ascii=False, indent=2)


# --- Prompt: шаблон, не готова вигадана відповідь ----------------------------


class TravelReplyArgs(BaseModel):
    """Валідовані параметри шаблону відповіді туристичному агенту-помічнику."""

    traveler_name: str = Field(min_length=1, max_length=100, description="Ім'я мандрівника для звернення")
    issue_summary: str = Field(min_length=3, max_length=300, description="Короткий опис ситуації чи питання")
    tone: Literal["формальний", "дружній", "стриманий"] = Field(
        default="дружній", description="Тон відповіді: формальний, дружній або стриманий"
    )


_OPENINGS = {
    "формальний": "Шановний(а) {name},",
    "дружній": "Привіт, {name}!",
    "стриманий": "{name},",
}


@mcp.prompt()
def travel_reply(traveler_name: str, issue_summary: str, tone: str = "дружній") -> str:
    """Шаблон відповіді мандрівнику з підставленими параметрами (не готова відповідь).

    Повертає текстовий каркас, який агент/LLM має доповнити фактами з
    tools (`get_weather`, `calculate_trip_cost`, `find_activities`,
    `search_knowledge`) — сам prompt нічого не вигадує і не викликає LLM.

    Args:
        traveler_name: Ім'я мандрівника, 1–100 символів.
        issue_summary: Короткий опис ситуації, 3–300 символів.
        tone: Один із: "формальний", "дружній", "стриманий" (типово "дружній").
    """
    try:
        validated = TravelReplyArgs(traveler_name=traveler_name, issue_summary=issue_summary, tone=tone)
    except ValidationError as exc:
        raise ValueError(_validation_error_message(exc)) from exc

    opening = _OPENINGS[validated.tone].format(name=validated.traveler_name)
    return (
        f"{opening}\n\n"
        f"Щодо вашого запиту: {validated.issue_summary}\n\n"
        "[Тут агент має додати конкретні факти з відповідних tools: "
        "get_weather / calculate_trip_cost / find_activities / search_knowledge.]\n\n"
        "З повагою,\nТуристична служба підтримки"
    )


if __name__ == "__main__":
    logger.info(
        "Запуск travel-support MCP сервера (stdio); knowledge=%s booking=%s",
        KNOWLEDGE_PATH,
        BOOKING_PATH,
    )
    mcp.run(transport="stdio")
