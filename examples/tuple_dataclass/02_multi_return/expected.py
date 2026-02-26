from typing import Any
from dataclasses import dataclass


@dataclass
class ParseColorResult:
    r: Any
    g: Any
    b: Any
    a: Any


def _parse_color(text):
    text = text.strip().lower()
    if text.startswith("#"):
        r = int(text[1:3], 16)
        g = int(text[3:5], 16)
        b = int(text[5:7], 16)
        a = 255
        return ParseColorResult(r = r, g = g, b = b, a = a)
    return ParseColorResult(r = 128, g = 128, b = 128, a = 255)
