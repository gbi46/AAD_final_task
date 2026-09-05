"""Захисні механізми агента: ліміт кроків, тайм-аут, детектор циклів.

Перенесено без змін з `Kravchuk_AAD_1/safety.py` (ДЗ1, Завд. 3). Для MAS
(S05) кожен спеціалізований агент отримує власний екземпляр `ExecutionGuard`
— так лічильники кроків, тайм-аут і історія викликів для одного агента не
змішуються з іншими (per-agent), без додаткової абстракції поверх дата-класу.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


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
