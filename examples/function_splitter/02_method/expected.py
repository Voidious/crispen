# After splitting: the statistics computation moves to a @staticmethod.
# (Exact split point and parameter list depend on free-variable analysis.)
class ReportGenerator:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _compute_totals(values, records, count, mean, median):
        peak = values[-1] if values else 0.0
        bottom = values[0] if values else 0.0
        variance = (
            sum((r.value - mean) ** 2 for r in records) / count if count > 0 else 0.0
        )
        std_dev = variance**0.5
        return {
            "count": count,
            "mean": mean,
            "median": median,
            "peak": peak,
            "bottom": bottom,
            "std_dev": std_dev,
        }

    def build_report(self, dataset):
        if not dataset:
            raise ValueError("dataset is empty")
        if len(dataset) < self.config.min_samples:
            raise ValueError("insufficient samples")
        records = [r for r in dataset if r.is_valid()]
        count = len(records)
        total = sum(r.value for r in records)
        mean = total / count if count > 0 else 0.0
        values = sorted(r.value for r in records)
        median = values[count // 2] if values else 0.0
        return ReportGenerator._compute_totals(values, records, count, mean, median)
