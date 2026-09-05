# AAD Final Task — Travel Support Multi-Agent System

Фінальне домашнє завдання курсу: production-подібна мультиагентна система (MAS)
на LangGraph з MCP-сервером, guardrails, HITL, observability, evals та
red-teaming. Цей README поки описує лише підготовчий стан (S01 — репозиторій
та середовище); архітектура MAS, MCP-сервер та інструкція запуску повного
сценарію буде додана на наступних кроках.

## Домен

Туристична координація: бюджет і бронювання, погодний маршрут та активності,
туристичні правила й FAQ. Мультиагентна система маршрутизує запит між
спеціалізованими агентами через supervisor-вузол:

| Роль | Відповідальність |
|---|---|
| `billing` | бюджет і бронювання (plan → execute → replan) |
| `tech` | погода, активності, операційна підтримка маршруту (ReAct) |
| `researcher` | туристичні правила та FAQ (Agentic RAG) |
| `general` | привітання, уточнення, запити поза доменом (без привілейованих tools) |

Компоненти перевикористовуються з попередніх домашніх завдань (ReAct-агент,
Plan-and-Execute, ChromaDB RAG, TrajectoryLogger, ExecutionGuard) — детальна
таблиця походження компонентів з'явиться на кроці S02.

## Середовище

- Python 3.12 (перевірено; сумісний діапазон 3.11–3.12).
- Один LLM-провайдер обирається через змінну середовища `LLM_PROVIDER`
  (за замовчуванням `gemini`, `GEMINI_MODEL=gemini-2.5-flash`) — див.
  `.env.example`. Модель не хардкодиться в коді.

### Встановлення

```bash
python3.12 -m venv .venv   # або: python3.12 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # заповнити реальний ключ локально, файл не комітиться
```

### Перевірка встановлення

```bash
python --version
pip check
```

Фактичні результати перевірки середовища (версії пакетів, `pip check`,
smoke-imports для MCP/LangGraph/ChromaDB/Pydantic) зафіксовано в
`codex_review/evidence/` поза цим репозиторієм (крок S01).

## Перевикористані компоненти (S02)

Джерело — попередні домашні завдання того ж автора (репозиторії поза цим
проєктом; шляхи нижче лише пояснюють походження, не є runtime-залежністю).

| Компонент | Походження | Зміна в цьому проєкті |
|---|---|---|
| `knowledge.py` (`KnowledgeBase`, `LocalHashEmbedding`, 10 документів) | ДЗ2, Завд. 3 (Agentic RAG) | Без змін бізнес-логіки — модуль вже був ізольований і відтворюваний. |
| `safety.py` (`ExecutionGuard`, `SafetyStop`) | ДЗ1, Завд. 3 | Без змін коду; для MAS кожен агент отримує окремий екземпляр — так лічильники кроків/тайм-аут/повтори лишаються per-agent без нової абстракції. |
| `trajectory_logger.py` (`TrajectoryLogger`, `save_trajectories`) | ДЗ1, Завд. 3 | Додано поле `agent_name` до кожного кроку; додано санітизацію значень, схожих на API-ключі, перед записом; ліміт довжини поля збільшено з жорстких 500 символів до 4000, щоб крок не перетворювався на урізане резюме. |
| `tools_legacy.py`: `get_weather`, `find_activities`, каталог `CITY_DATA` | ДЗ1, Завд. 1 | Каталог міст об'єднано з ДЗ2 (Варшава додана до Києва/Львова/Одеси/Берліна/Парижа — конфліктів значень немає); винесено `*_core`-функції окремо від LangChain `@tool`-обгортки. `convert_currency` з ДЗ1 не перенесено — не входить у п'ять обов'язкових MCP-tools. |
| `tools_legacy.py`: `calculate_trip_cost`, `search_knowledge`, `book_hotel` | ДЗ2, Завд. 1/3 | `book_hotel` розширено: детермінований `idempotency_key` з параметрів бронювання — повторний виклик з тими самими аргументами не створює дублікат, а повертає наявний запис; сховище (`bookings.json`) переживає рестарт процесу (перевірено). |
| `tests/test_tools.py`, `test_knowledge.py`, `test_safety.py` | ДЗ1 + ДЗ2 | Перенесено й об'єднано під нові імена (`domain_tools` → `tools_legacy`); додано тест ідемпотентності бронювання. Сховище кожного тесту ізольоване через `tempfile`. |
| `tests/test_trajectory_logger.py` | новий | Раніше `TrajectoryLogger` перевірявся лише опосередковано через `agent.py`; тепер є прямі тести на `agent_name`, санітизацію та відсутність урізання довгих кроків. |

### Запуск тестів

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Станом на S02: 16 тестів (5 підтестів), усі проходять.

## MCP-сервер (S03)

`mcp_server.py` — транспортний шар над `tools_legacy.py`/`knowledge.py`
через офіційний MCP SDK (`mcp.server.fastmcp.FastMCP`, stdio entry point;
не плутати з окремим пакетом `fastmcp` — API різняться).

- **5 tools**: `get_weather`, `calculate_trip_cost`, `find_activities`,
  `search_knowledge`, `book_hotel` — кожен повторно валідує аргументи тими
  самими Pydantic-схемами з `tools_legacy.py` (сервер не довіряє
  клієнтській валідації) і викликає спільну `*_core`-функцію.
- **2 read-only resources**: `faq://general` (статичний туристичний FAQ,
  без RAG-пошуку) та `travel://cities` (каталог підтримуваних міст із
  демонстраційними даними CITY_DATA).
- **1 prompt**: `travel_reply(traveler_name, issue_summary, tone)` —
  повертає текстовий шаблон із підставленими параметрами (не готову
  вигадану LLM-відповідь); `tone` валідується як один із трьох варіантів.
- Логи йдуть у stderr (`logging.basicConfig(..., stream=sys.stderr)`),
  stdout лишається виключно протоколу. Шляхи сховищ — `MCP_KNOWLEDGE_PATH`
  / `MCP_BOOKING_PATH` (типово відносно розташування самого файлу, не cwd
  процесу, що його запустив).
- Невалідні аргументи, невідоме місто чи збій ChromaDB підіймають
  `ValueError`/`RuntimeError`, які MCP перетворює на
  `CallToolResult(isError=True, ...)` — перевірено як напряму, так і через
  повний stdio-протокол (`ClientSession` у окремому subprocess): успішний
  виклик має `isError=False`, невалідний — `isError=True` з описом
  причини, storage не змінюється.

### Запуск сервера

```bash
python mcp_server.py
```

Для ручної перевірки: `npx @modelcontextprotocol/inspector python mcp_server.py`.

## MCP-тести (S04)

`tests/test_mcp_server.py` — 13 async-тестів через справжній MCP client
API (`mcp.ClientSession` + `mcp.client.stdio.stdio_client`), а не прямі
виклики Python-функцій: кожен тест піднімає `mcp_server.py` як окремий
subprocess з ізольованим tempdir-сховищем (`tmp_path`) і перевіряє
transport-шар разом із бізнес-логікою.

Покриття: `list_tools` (точний набір і схеми, включно з required/default),
`get_weather`/`calculate_trip_cost` на fixtures, `find_activities` і
`search_knowledge` (document IDs у відповіді), невалідні `city`/`days`/
booking-параметри (помилка, storage не змінюється), ідемпотентність
`book_hotel` (повтор не дублює запис), `list_resources`/`read_resource`
для обох URI (повторне читання не змінює вміст), `list_prompts`/
`get_prompt` (шаблон, параметри, невалідний `tone` відхиляється), окремий
stdio smoke-тест з `asyncio.wait_for(..., timeout=20)` — підтверджує
відсутність зависання subprocess. Жоден тест не потребує LLM API-ключа.

```bash
python -m pytest tests/ -v         # усі тести, ~16 с (MCP-тести стартують subprocess)
python -m pytest tests/test_mcp_server.py -v  # лише MCP
```

Станом на S04: 29 тестів (5 підтестів), усі проходять.

## Статус

Виконано: S01 (репозиторій та середовище), S02 (перевикористані
компоненти й контракти), S03 (MCP-сервер: tools, resources, prompt), S04
(MCP-тести через async client). Наступний крок — S05 (LangGraph
supervisor і агенти) за планом `claude_plan/`.
