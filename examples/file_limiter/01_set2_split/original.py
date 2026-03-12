def process(records):
    results = []
    for r in records:
        name = r["name"].strip().lower()
        results.append(name)
    return results
