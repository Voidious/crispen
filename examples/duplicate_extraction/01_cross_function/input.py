def generate_report(period):
    config = load_config()
    db = Database(config.db_url)
    formatter = ReportFormatter(config.format)
    data = db.query_for_period(period)
    return formatter.format(data)


def generate_summary(region):
    config = load_config()
    db = Database(config.db_url)
    formatter = ReportFormatter(config.format)
    totals = db.query_totals_for_region(region)
    return formatter.summarize(totals)
