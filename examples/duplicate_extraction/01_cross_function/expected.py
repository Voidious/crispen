# After extraction: the shared 3-statement setup block becomes a helper.
# (Exact whitespace around the inserted helper may vary slightly.)
def _setup_report_resources():
    config = load_config()
    db = Database(config.db_url)
    formatter = ReportFormatter(config.format)
    return config, db, formatter


def generate_report(period):
    config, db, formatter = _setup_report_resources()
    data = db.query_for_period(period)
    return formatter.format(data)


def generate_summary(region):
    config, db, formatter = _setup_report_resources()
    totals = db.query_totals_for_region(region)
    return formatter.summarize(totals)
