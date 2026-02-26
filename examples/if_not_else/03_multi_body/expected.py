# Both branches have multiple statements.
# All statements in each branch must be correctly swapped — not just the first.
def save_record(record, db, logger):
    if record.is_valid():
        row = db.insert(record)
        logger.info("Saved record: %s", record.id)
        metrics.increment("records.saved")
        return row
    else:
        logger.warning("Skipping invalid record: %s", record.id)
        metrics.increment("records.invalid")
        return None
