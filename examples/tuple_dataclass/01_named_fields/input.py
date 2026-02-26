def _parse_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    host = data["host"]
    port = int(data["port"])
    db = data.get("database", "default")
    timeout = float(data.get("timeout", 30))
    return (host, port, db, timeout)
