from typing import Any
from dataclasses import dataclass


@dataclass
class ParseConfigResult:
    host: Any
    port: Any
    db: Any
    timeout: Any


def _parse_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    host = data["host"]
    port = int(data["port"])
    db = data.get("database", "default")
    timeout = float(data.get("timeout", 30))
    return ParseConfigResult(host = host, port = port, db = db, timeout = timeout)
