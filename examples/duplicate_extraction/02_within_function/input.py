def sync_tables(source_url, dest_url):
    # Sync first table
    src = connect(source_url)
    dest = connect(dest_url)
    src.ping()
    dest.ping()
    rows = src.fetch_all("orders")
    dest.bulk_insert("orders", rows)
    # Sync second table
    src = connect(source_url)
    dest = connect(dest_url)
    src.ping()
    dest.ping()
    rows = src.fetch_all("products")
    dest.bulk_insert("products", rows)
