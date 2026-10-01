"""
Tests the weather MCP server the same way LangGraph will use it in Step 2:
as a real MCP client that starts the server and talks to it over stdio.

Run from the project folder:
    python mcp_servers/test_weather_server.py
"""

import asyncio
import sys
import warnings
from pathlib import Path

# Harmless warning from a library the mcp package uses (Python 3.14)
warnings.filterwarnings("ignore", message=".*has an incomplete definition.*")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252 (°C would fail)

SERVER_FILE = Path(__file__).resolve().parent / "weather_server.py"

# (city, days) - the last two check that errors come back as friendly messages
TEST_CASES = [
    ("Jaipur", 3),
    ("Paris, FR", 5),
    ("Tokyo", 7),          # asks for more days than the free forecast has
    ("Atlantisxyz", 2),    # a city that doesn't exist
]


async def main():
    # 1. Tell the client how to start the server (as a separate Python program)
    server = StdioServerParameters(command=sys.executable, args=[str(SERVER_FILE)])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            # 2. The MCP "handshake"
            await session.initialize()

            # 3. Ask the server which tools it offers
            tools = await session.list_tools()
            print("Tools offered by the server:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description.strip().splitlines()[0]}")
                print(f"    inputs: {list(tool.inputSchema.get('properties', {}))}")

            # 4. Call the tool for each test case
            for city, days in TEST_CASES:
                print(f"\n=== get_weather_forecast(city={city!r}, days={days}) ===")
                result = await session.call_tool("get_weather_forecast", {"city": city, "days": days})
                print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
