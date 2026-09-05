"""Порожній conftest.py в корені — забезпечує, що pytest (у будь-якій формі
запуску: `pytest`, `python -m pytest`) додає корінь проєкту в sys.path, щоб
тести `tests/` могли імпортувати флет-модулі (`tools_legacy`, `knowledge`,
`safety`, `trajectory_logger`) без встановлення пакета.
"""
