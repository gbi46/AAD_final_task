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

## Статус

Виконано: S01 (репозиторій та середовище). Наступні кроки — S02
(перевикористані компоненти) і далі за планом `claude_plan/`.
