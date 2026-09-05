# AAD Final Task — Travel Support Multi-Agent System

Фінальне домашнє завдання курсу: production-подібна мультиагентна система (MAS)
на LangGraph з MCP-сервером, guardrails, HITL, observability, evals та
red-teaming. Базовий рівень (MAS + MCP) у розробці; підключення MCP до
агентів, guardrails, HITL, observability/evals/red-team і бонусний рівень
(CrewAI) буде додано на наступних кроках — див. розділ «Статус».

## Domain

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

## Environment

- Python 3.12 (перевірено; сумісний діапазон 3.11–3.12).
- Один LLM-провайдер обирається через змінну середовища `LLM_PROVIDER`
  (за замовчуванням `gemini`, `GEMINI_MODEL=gemini-2.5-flash`) — див.
  `.env.example`. Модель не хардкодиться в коді.

### Installation

```bash
python3.12 -m venv .venv   # або: python3.12 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # заповнити реальний ключ локально, файл не комітиться
```

### Installation Check

```bash
python --version
pip check
```

Фактичні результати перевірки середовища (версії пакетів, `pip check`,
smoke-imports для MCP/LangGraph/ChromaDB/Pydantic) зафіксовано в
`codex_review/evidence/` поза цим репозиторієм (крок S01).

## Reused Components (S02)

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

### Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Станом на S02: 16 тестів (5 підтестів), усі проходять.

## MCP Server (S03)

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

### Running the Server

```bash
python mcp_server.py
```

Для ручної перевірки: `npx @modelcontextprotocol/inspector python mcp_server.py`.

## MCP Tests (S04)

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

## MAS in LangGraph (S05)

`mas_langgraph.py` — supervisor-вузол маршрутизує запит між чотирма
спеціалізованими агентами. Локальні валідовані wrappers з `tools_legacy.py`
викликаються напряму; підключення того самого набору tools через MCP —
окремий крок S06.

- **Supervisor**: `RouteDecision(action, reasoning)` — live-режим викликає
  `with_structured_output(RouteDecision)` з few-shot прикладами і фолбеком
  на `general` при невалідному/зависшому structured output; demo-режим
  (`DemoSupervisorRouter`) — детермінований, ключовими словами, але
  проходить той самий граф. Маршрутизація читає `HumanMessage` (ціль
  користувача) чи явний `handoff_context`, а не останнє `ToolMessage`.
- **billing** — підграф `planner → executor → replanner`, перенесений і
  адаптований з `Kravchuk_AAD_2/plan_execute.py` (без persistence/HITL —
  це S07/S09; `book_hotel` тут не викликається автоматично). Реальний
  replan: якщо орієнтовний бюджет перевищує 500 EUR, replanner додає крок
  `search_knowledge` з порадами економії — план справді зростає з 1 до 2
  кроків, це не позначка, а фактична зміна залишку плану.
- **tech** — ReAct `agent → tools → agent`, перенесений і адаптований з
  `Kravchuk_AAD_1/agent.py` (`get_weather`, `find_activities`). Детектор
  повторюваних викликів (`ExecutionGuard.max_repeats`) контрольовано
  зупиняє граф на сценарії "повторюй ...". Реальний handoff: якщо запит
  також стосується бюджету, tech передає `handoff_context` — supervisor
  на наступному візиті одразу маршрутизує до billing (без повторної
  LLM-класифікації), обмежено `MAX_HANDOFFS=1`.
- **researcher** — LLM (чи demo-евристика) вирішує, чи потрібен
  `search_knowledge`; відповідь цитує ID документів (`[travel-XX]`).
  Порівняння: довідковий запит (RAG потрібен) проти арифметичного
  бюджетного запиту (RAG не потрібен) — різні гілки супервізора.
- **general** — відповідь/уточнення без жодного tool-виклику.
- **Бюджети**: per-agent `ExecutionGuard` (max_steps/timeout/max_repeats)
  для billing/tech/researcher — обмежують і вкладені цикли всередині
  підграфів, не лише зовнішній супервізорний цикл. Глобальний
  `MAX_SUPERVISOR_VISITS=8` зупиняє граф навіть якщо конкретний агент
  ніколи не встановлює `completed`. `safety.call_with_timeout` (новий у
  S05) для LLM-викликів у live-режимі: на відміну від
  `ExecutionGuard.check_limits()` (перевіряє час до/після виклику), тут
  абонент дійсно перестає чекати на межі тайм-ауту, а не лише дізнається
  про перевищення вже після того, як виклик сам завершився.
- **Reducers стану**: `messages` — вбудований `add_messages` (дедуп за
  id), `trajectory` — власний reducer, що додає нові кроки спільного
  `TrajectoryLogger` без дублювання, `results` — накопичення
  (`operator.add`), `plan` — явна заміна (без reducer, останній запис
  перемагає).

```bash
python -m pytest tests/test_mas_langgraph.py -v   # локальні wrappers, без MCP
```

Станом на S05: 42 тести (5 підтестів), усі проходять.

## MCP Integration with Agents (S06)

`MASApp.connect_mcp()` піднімає `mcp_server.py` (S03) як окремий subprocess
через `MultiServerMCPClient` (stdio, абсолютний шлях до файлу сервера,
поточний `sys.executable` — не системний `python` з PATH) і замінює
локальні tools у `self.tools_by_name` на MCP-tools з тими самими іменами.
Виконання йде через ОДИН спільний dispatcher (`MASApp.dispatch_tool`) — той
самий метод обслуговує і локальні LangChain-обгортки (за замовчуванням, без
`connect_mcp()`), і MCP-tools; billing/tech/researcher не знають, звідки
саме виконався виклик.

- `MultiServerMCPClient` відкриває нову stdio-сесію на кожен виклик tool,
  resource чи prompt (задокументована поведінка адаптера) — довготривалого
  з'єднання тримати не потрібно, ціна — subprocess-старт на кожен виклик
  (~1 с через ChromaDB seeding у `mcp_server.py`, прийнятно для демо).
- **`handle_tool_errors=False`** при створенні клієнта: за замовчуванням
  адаптер ковтає `CallToolResult(isError=True)` і повертає текст помилки
  як звичайний "успішний" рядок (щоб ReAct-агент міг самовиправитись) —
  саме той фальшивий успіх, якого явно вимагає уникати план. Із `False`
  таку помилку MCP-сервера піднімає як виключення, dispatcher ловить його
  й формує `ToolDispatchError` — знайдено емпірично під час перевірки.
- MCP-tool повертає список content-блоків (`[{"type": "text", "text": ...}]`),
  а не голий рядок — `dispatch_tool` нормалізує це до плоского тексту
  (інакше агенти бачили б Python-репр списку словників).
- **researcher**: спершу перевіряє статичний resource `faq://general`
  (дешевше за RAG; самопосилальний запис FAQ, що сам відсилає до
  `search_knowledge`, виключено з матчингу — інакше слово "правила"
  хибно перехоплювало б запити на кшталт "Які правила відвідування
  Лувру?", які насправді потребують справжнього RAG-пошуку).
- **general**: формує відповідь через prompt `travel_reply` — шаблон із
  підставленими параметрами, а не готова вигадана LLM-відповідь.
- Кожен `tool_dispatch`/`tool_dispatch_error` запис у trajectory містить
  `tool_call_id`, аргументи, тривалість і `agent_name` — і для успіху, і
  для помилки.

```bash
python mas_langgraph.py   # demo-прогін 7 сценаріїв, tools через реальний MCP subprocess
python -m pytest tests/test_mas_mcp_integration.py -v   # реальні MCP-виклики (кожен тест — окремий subprocess)
```

Демонстрації (усі виконано, ≥2 звертаються до subprocess MCP — фактично всі 9
tool-викликів пройшли через нього): «Оціни бюджет Львова на 3 дні для двох»
(у складеному запиті з погодою) → tech → handoff → billing; «У Львові дощ:
перевір погоду і підбери культурні активності» → tech; «Які правила
відвідування Лувру?» → researcher (RAG через MCP). Повний `trajectory.json`
з усіма demo-запусками (agent_name на кожному кроці) зберігається у
gitignored `runtime/` (не контрольний артефакт здачі — той буде
підготовлений окремо в S07).

Станом на S06: 49 тестів (5 підтестів), усі проходять; 5 з них (`test_mas_mcp_integration.py`)
реально піднімають subprocess MCP-сервера.

## Durable State (S07)

MAS зберігає стан через файловий `AsyncSqliteSaver` (`langgraph.checkpoint.sqlite.aio`
— граф викликається через `ainvoke`, а синхронний `SqliteSaver` явно не
підтримує async-методи) з `thread_id` на кожну сесію. `MASApp(checkpointer=...,
interrupt_after=[...])` — той самий граф, що й у S05/S06, лише з двома новими
опційними параметрами.

`persistence_demo.py` — відтворювана демонстрація рівно того, що вимагає
план: два ОКРЕМІ Python-процеси (не два виклики в одному інтерпретаторі):

```bash
python persistence_demo.py pause    # процес А: 1 крок (маршрутизація), пауза, закриття з'єднання
python persistence_demo.py resume   # процес Б: нове з'єднання, той самий thread_id, довиконання
```

- **Процес А**: `interrupt_after=["supervisor"]` зупиняє граф одразу після
  ОДНОГО виконаного кроку — ще до того, як billing реально викличе tool;
  знімок стану (`plan=[]`, `current_step=0`, `results=[]`) підтверджує це.
- **Процес Б**: нове з'єднання до того самого `agent_state.db` і
  `thread_id`, `resume()` (`None` як вхід — стандартний спосіб LangGraph
  продовжити з checkpoint) доводить billing до кінця. Порівняння
  `plan`/`current_step`/`results` до і після: `current_step: 0 -> 2`,
  `results` містить РІВНО стільки записів, скільки кроків у плані (не
  вдвічі більше) — доводить відсутність повторного виконання вже
  зробленого кроку.
- **Ізоляція сесій**: інший `thread_id` (`persistence-demo-isolated`) не
  бачить жодного стану першого — перевірено в обох процесах.
- **Цілісність SQLite без WAL**: `SqliteSaver`/`AsyncSqliteSaver` вмикають
  `PRAGMA journal_mode=WAL`, тому свіжі записи спершу лежать в окремому
  `-wal` файлі. Після фази `resume` виконується
  `PRAGMA wal_checkpoint(TRUNCATE)` + повернення в `journal_mode=DELETE`
  — доставлений `agent_state.db` самодостатній, без потреби у
  supplementary `-wal`/`-shm` файлах (перевірено: `PRAGMA integrity_check`
  → `ok`, sidecar-файли відсутні).
- `trajectory.json` (контрольний артефакт) об'єднує обидві фази; кожен
  крок має `agent_name`.

```bash
python -m pytest tests/test_persistence.py -v
```

5 тестів — кожен реально запускає `persistence_demo.py pause`/`resume` як
окремий `subprocess.run([sys.executable, ...])` (не прямий виклик функцій
у тому самому інтерпретаторі), з ізольованими шляхами через
`PERSISTENCE_DB_PATH`/`PERSISTENCE_KNOWLEDGE_PATH`/`PERSISTENCE_TRAJECTORY_PATH`/
`PERSISTENCE_RUNTIME_DIR` — жоден тест не чіпає справжні контрольні
артефакти в корені.

Станом на S07: 54 тести (5 підтестів), усі проходять.

## Base-Level Control Artifacts

`agent_state.db`, `chroma_db/`, `trajectory.json` у корені репозиторію —
контрольні артефакти здачі базового рівня (синтетичний billing-запит,
demo-режим, без реальних секретів чи персональних даних), згенеровані
`persistence_demo.py` і після цього заморожені (writers закрито,
`agent_state.db` явно checkpoint'нуто — без залежності від WAL). Звичайні
runtime-сесії розробки (S05/S06 demo-прогони) ізольовано в gitignored
`runtime/` і в артефакти здачі не потрапляють.

## Status

Виконано: S01–S06 (репозиторій, перевикористані компоненти, MCP-сервер і
тести, LangGraph supervisor з агентами billing/tech/researcher/general,
підключення MCP до агентів), S07 (durable state: файловий SqliteSaver,
pause/resume між процесами, ізоляція сесій, контрольні артефакти здачі
базового рівня). Це завершує базовий рівень (до 65 балів). Наступний
крок — S08 (guardrails: input/output/tool/rate-limit) за планом
`claude_plan/`.
