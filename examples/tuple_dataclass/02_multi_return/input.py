def _parse_color(text):
    text = text.strip().lower()
    if text.startswith("#"):
        r = int(text[1:3], 16)
        g = int(text[3:5], 16)
        b = int(text[5:7], 16)
        a = 255
        return (r, g, b, a)
    return (128, 128, 128, 255)
