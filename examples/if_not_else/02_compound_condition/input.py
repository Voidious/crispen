# Compound boolean condition wrapped in `not (...)`.
# The parenthesized expression is preserved as-is after the flip.
def validate_bounds(x, y, width, height):
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Point ({x}, {y}) is out of bounds")
    else:
        return True
