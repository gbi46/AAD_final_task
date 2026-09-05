"""CLI для вимірювання CrewAI: час, токени та події агентів і MCP-tools."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    query = sys.argv[1]
    runtime_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else (_ROOT / "runtime" / "crewai_bench")

    if not os.getenv("OPENROUTER_API_KEY"):
        print(json.dumps({"error": "OPENROUTER_API_KEY відсутній"}))
        raise SystemExit(1)

    from crewai.events import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

    calls: list[dict[str, str]] = []

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def _on_tool_usage(source: object, event: ToolUsageStartedEvent) -> None:  # noqa: ANN001
        calls.append({"agent_role": event.agent_role or "", "tool_name": event.tool_name})

    import mas_crewai as mc

    result = mc.run_query_with_metrics(query, runtime_dir)
    result["tool_calls"] = calls
    result["agents_used"] = list(dict.fromkeys(c["agent_role"] for c in calls if c["agent_role"]))
    result["tools_called"] = [c["tool_name"] for c in calls]

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
