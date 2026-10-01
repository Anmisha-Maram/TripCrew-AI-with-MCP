"""
Tests all 3 MCP servers together, through MultiServerMCPClient - exactly the way
the agents in backend.py use them (tools/mcp_tools.py).

Run from the project folder:
    python mcp_servers/test_mcp_tools.py

Uses: 1 OpenWeather call, 1 AviationStack request (free plan: 100/month), 1 Tavily credit.
No Groq tokens.
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "tools" can be imported

from tools.mcp_tools import call_mcp_tool, load_mcp_tools
from tools.tavily_tool import format_mcp_search_result

PREVIEW_CHARS = 700  # how much of each answer to print


def main():
    # 1. Connect to every server and list its tools
    print("Connecting to MCP servers...\n")
    start = time.time()
    tools = load_mcp_tools()
    print(f"\nConnected to {len(tools)} of 3 servers in {time.time() - start:.1f}s\n")

    # 2. Call one tool on each server
    tests = [
        ("weather", "get_weather_forecast", {"city": "Goa, IN", "days": 2}),
        ("flights", "search_flights", {"query": "Plan a 4 days trip from Delhi to Goa"}),
        ("tavily", ["tavily_search", "tavily-search"],
         {"query": "Best hotels in Goa", "max_results": 3, "search_depth": "basic"}),
    ]

    for server, tool, arguments in tests:
        print(f"=== {server} -> {tool if isinstance(tool, str) else tool[0]}({arguments}) ===")
        start = time.time()
        try:
            answer = call_mcp_tool(server, tool, arguments)
            if server == "tavily":
                # Same step as the hotel agent: raw JSON -> short text
                raw_size = len(answer)
                answer = format_mcp_search_result(answer)
                print(f"(Tavily sent {raw_size} characters of JSON; tidied to {len(answer)})")
            print(answer[:PREVIEW_CHARS] + (" ...(cut for display)" if len(answer) > PREVIEW_CHARS else ""))
            print(f"--- OK in {time.time() - start:.1f}s, {len(answer)} characters\n")
        except Exception as e:
            print(f"--- FAILED: {e}\n")


if __name__ == "__main__":
    main()
