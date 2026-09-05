"""Захисні механізми агента: ліміт кроків, тайм-аут, детектор циклів.

`ExecutionGuard`/`SafetyStop` перенесено без змін з `Kravchuk_AAD_1/safety.py`
(ДЗ1, Завд. 3). Для MAS (S05) кожен спеціалізований агент отримує власний
екземпляр `ExecutionGuard` — так лічильники кроків, тайм-аут і історія
викликів для одного агента не змішуються з іншими (per-agent), без
додаткової абстракції поверх дата-класу.

`call_with_timeout` — новий для S05: `ExecutionGuard.check_limits()` лише
порівнює час ПІСЛЯ того, як виклик уже завершився (чи перед наступною
ітерацією), тому не рятує від зависання одного конкретного виклику (напр.,
LLM-запиту, що ніколи не повертається). `call_with_timeout` справді
повертає керування абоненту на межі тайм-ауту.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


class SafetyStop(RuntimeError):
    """Контрольована зупинка виконання."""


@dataclass
class ExecutionGuard:
    max_steps: int = 10
    timeout_seconds: float = 60.0
    max_repeats: int = 3
    started_at: float = field(default_factory=time.monotonic)
    recent_calls: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.started_at = time.monotonic()
        self.recent_calls.clear()

    def check_limits(self, step_count: int) -> None:
        if step_count >= self.max_steps:
            raise SafetyStop(f"Досягнуто ліміт кроків ({self.max_steps}).")
        if time.monotonic() - self.started_at >= self.timeout_seconds:
            raise SafetyStop(f"Перевищено загальний тайм-аут ({self.timeout_seconds:g} с).")

    def register_call(self, name: str, arguments: dict) -> None:
        payload = json.dumps([name, arguments], ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.recent_calls.append(fingerprint)
        if len(self.recent_calls) >= self.max_repeats:
            window = self.recent_calls[-self.max_repeats :]
            if len(set(window)) == 1:
                raise SafetyStop(
                    f"Виявлено цикл: {name} з однаковими аргументами викликано "
                    f"{self.max_repeats} рази поспіль."
                )


def call_with_timeout(
    func: Callable[_P, _T], timeout_seconds: float, *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """Виконати `func` з реальним обмеженням часу очікування абонента.

    На відміну від `ExecutionGuard.check_limits()` (перевіряє час до/після
    виклику), тут абонент дійсно перестає чекати рівно через
    `timeout_seconds` і отримує `SafetyStop`, навіть якщо сам виклик ще не
    завершився. Чесне застереження: Python не має способу примусово вбити
    потік, тому фоновий потік із зависшим викликом продовжує існувати у
    фоні — ми лише гарантуємо, що абонент не блокується назавжди.

    Важливо: executor НЕ використовується як context manager — `__exit__`
    типового `with ThreadPoolExecutor()` викликає `shutdown(wait=True)`,
    який сам блокується до завершення того самого зависшого потоку і
    звів би нанівець весь сенс тайм-ауту. Тому shutdown викликається
    окремо з `wait=False`.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        raise SafetyStop(
            f"Виклик перевищив тайм-аут ({timeout_seconds:g} с) і абонент його не чекає далі."
        ) from exc
    finally:
        executor.shutdown(wait=False)
