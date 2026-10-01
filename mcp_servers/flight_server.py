"""
Flight MCP server for TripCrew AI.

This server does not contain any flight logic of its own. It simply offers the
existing search_flights() from tools/flight_tool.py (AviationStack via APILayer)
as an MCP tool, so agents can use it through the Model Context Protocol.

Run directly (the client normally starts it for you):
    python mcp_servers/flight_server.py

Like the weather server, this talks over stdin/stdout ("stdio"), so never print() here.
"""

import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Make "tools/..." importable, and load .env, no matter which folder we are started from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Harmless warning from a library the mcp package uses (Python 3.14); hide it to keep logs clean
warnings.filterwarnings("ignore", message=".*has an incomplete definition.*")

from mcp.server.fastmcp import FastMCP

from tools.flight_tool import search_flights as aviationstack_search  # the existing tool

mcp = FastMCP("flights", log_level="WARNING")


@mcp.tool()
def search_flights(query: str, limit: int = 3) -> str:
    """
    Find live flights for a trip described in plain English.

    Args:
        query: The trip request, e.g. "Plan a 4 day trip from Delhi to Goa".
               Cities, countries and airport codes (DEL, JFK) are understood.
        limit: How many flights to return (default 3).

    Returns one short line per flight: flight number, airline, route,
    departure and arrival times, and status. Shows live flight status, not ticket prices.
    """
    return aviationstack_search(query, limit=limit)


if __name__ == "__main__":
    mcp.run()  # default transport is "stdio"
