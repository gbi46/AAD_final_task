"""Ізоляція тестів від телеметрії та локальних налаштувань трасування."""

import os

# Установлюється до імпортів агентів і успадковується MCP-підпроцесами.
# load_dotenv не замінює ці значення локальною конфігурацією користувача.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_DISABLED"] = "true"
