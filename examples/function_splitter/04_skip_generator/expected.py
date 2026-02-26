# Generator functions (containing yield) are never split — crispen skips them.
def stream_csv_records(filepath, batch_size=100):
    with open(filepath) as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
