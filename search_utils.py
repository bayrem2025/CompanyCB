from ddgs import DDGS

def search_web(query, max_results=5):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "No Title"),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        print(f"Search error: {e}")
    return results
