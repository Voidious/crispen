# After splitting: the summary-building tail references self.config and
# self.results, so the helper is a regular instance method (not @staticmethod).
class MetricsAggregator:
    def __init__(self, config):
        self.config = config
        self.results = []

    def aggregate(self, raw_events):
        counts = {}
        totals = {}
        for event in raw_events:
            key = event["type"]
            counts[key] = counts.get(key, 0) + 1
            totals[key] = totals.get(key, 0.0) + event.get("value", 0.0)
        return self._build_summary(counts, totals)


    def _build_summary(self, counts, totals):
        summary = {}
        for key in counts:
            avg = totals[key] / counts[key]
            limit = self.config.limits.get(key, float("inf"))
            flagged = avg > limit
            label = self.config.labels.get(key, key)
            summary[key] = {
                "count": counts[key],
                "total": totals[key],
                "avg": avg,
                "limit": limit,
                "flagged": flagged,
                "label": label,
            }
        self.results.append(summary)
        return summary
