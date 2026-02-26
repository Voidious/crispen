# Async functions are never split — crispen skips them.
async def fetch_all_pages(session, base_url):
    page = 0
    results = []
    while True:
        url = f"{base_url}?page={page}"
        response = await session.get(url)
        data = await response.json()
        if not data.get("items"):
            break
        results.extend(data["items"])
        page += 1
    return results
