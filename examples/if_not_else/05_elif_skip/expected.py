# elif chains cannot be safely flipped — crispen skips them.
def classify_score(score):
    if not score >= 90:
        return "fail"
    elif score < 100:
        return "pass"
    else:
        return "perfect"
