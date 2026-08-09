def _summarize_pending_report():
    rows = fetch_pending_rows()
    grouped = group_by_owner(rows)
    summary = format_summary(grouped)
    return summary
