from tavily import TavilyClient
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key = os.getenv("TAVILY_API_KEY")
)

def format_search_results(results):
    """Turns Tavily results into short text: title, link and a 300-character snippet each."""
    lines = []

    for i, r in enumerate(results,1):
        title   = r.get("title","Unknown")
        url     = r.get("url","")
        snippet = (r.get("content") or "").strip()

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0]+ "..."
        lines.append(f"{i}, **{title}**\n {url}\n {snippet}")

    return "\n\n".join(lines)

def format_mcp_search_result(raw_text):
    """
    Tavily's MCP server answers with raw JSON (scores, nulls, long content).
    Turn it into the same short format as tavily_search() to save Groq tokens.
    """
    try:
        data = json.loads(raw_text)
        return format_search_results(data.get("results", []))
    except (ValueError, AttributeError):
        return raw_text  # not JSON: use it as it is

def tavily_search(query):
    response = client.search(
        query= query,
        max_results= 5
    )

    return format_search_results(response["results"])
