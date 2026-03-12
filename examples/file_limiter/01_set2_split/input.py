def process(records):
    results = []
    for r in records:
        name = r["name"].strip().lower()
        results.append(name)
    return results


def normalize_name(text):
    return text.strip().lower()


def validate_record(record):
    return "name" in record and "value" in record
