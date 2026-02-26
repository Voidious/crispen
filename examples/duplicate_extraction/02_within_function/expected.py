# After extraction: the 4-statement connection setup becomes a helper.
# (Exact whitespace around the inserted helper may vary slightly.)
def _open_connections(source_url, dest_url):
    src = connect(source_url)
    dest = connect(dest_url)
    src.ping()
    dest.ping()
    return src, dest


def sync_tables(source_url, dest_url):
    # Sync first table
    src, dest = _open_connections(source_url, dest_url)
    rows = src.fetch_all("orders")
    dest.bulk_insert("orders", rows)
    # Sync second table
    src, dest = _open_connections(source_url, dest_url)
    rows = src.fetch_all("products")
    dest.bulk_insert("products", rows)
