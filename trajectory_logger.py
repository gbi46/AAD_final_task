"""JSON-траєкторії MAS з agent_name, вузлами й викликами інструментів.

Перед записом маскуються API-ключі та PII. Максимальна довжина поля —
4000 символів; довші значення обрізаються."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guardrails import output_guardrail

_SENSITIVE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.]{10,}"),
]
_REDACTED = "***REDACTED***"


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    redacted, _pii_found = output_guardrail(redacted)  # PII, не лише API-ключі
    return redacted


def _sanitize(value: Any) -> Any:
    """Рекурсивно маскує рядки, схожі на API-ключі, у будь-якій вкладеній структурі."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


class TrajectoryLogger:
    def __init__(self, run_id: str, max_field_chars: int = 4000):
        self.run_id = run_id
        self.max_field_chars = max_field_chars
        self.steps: list[dict[str, Any]] = []
        self.started_at = time.monotonic()

    def log(
        self,
        node: str,
        input_data: Any,
        output_data: Any,
        tool_calls: list | None = None,
        duration_ms: int = 0,
        agent_name: str = "supervisor",
    ) -> None:
        self.steps.append(
            {
                "step_number": len(self.steps) + 1,
                "agent_name": agent_name,
                "node_name": node,
                "input": _redact_text(str(input_data)[: self.max_field_chars]),
                "output": _redact_text(str(output_data)[: self.max_field_chars]),
                "tool_calls": _sanitize(tool_calls or []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": duration_ms,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.started_at) * 1000),
            "trajectory": self.steps,
        }


def save_trajectories(runs: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
