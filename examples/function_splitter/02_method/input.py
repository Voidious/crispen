class ReportGenerator:
    def __init__(self, config):
        self.config = config

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
