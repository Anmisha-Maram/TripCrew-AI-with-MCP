"""
Connects TripCrew AI to its MCP servers with MultiServerMCPClient
(from langchain-mcp-adapters).

    weather  -> mcp_servers/weather_server.py   (runs on this computer, "stdio")
    flights  -> mcp_servers/flight_server.py    (runs on this computer, "stdio")
    tavily   -> Tavily's hosted MCP server      (on the internet, "streamable_http")

Agents use one function:

    call_mcp_tool("flights", "search_flights", {"query": "Delhi to Goa"})

MCP tools are "async" (they run with asyncio), but our LangGraph graph,
FastAPI endpoints and test.py are normal "sync" code. call_mcp_tool() bridges
the two: it runs the async call and waits for the answer, so the rest of the
project does not need to change.
"""

import asyncio
import os
import sys
import threading
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Harmless warning from a library the mcp package uses (Python 3.14)
warnings.filterwarnings("ignore", message=".*has an incomplete definition.*")

from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"


# =========================================================
# The 3 MCP servers
# =========================================================

def get_connections() -> dict:
    """How to reach each MCP server."""
    connections = {
        # Local servers: the client starts them as small background Python programs
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(PROJECT_ROOT / "mcp_servers" / "weather_server.py")],
        },
        "flights": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(PROJECT_ROOT / "mcp_servers" / "flight_server.py")],
        },
    }

    # Hosted server: the API key is read from .env when the app runs (never written in code)
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        connections["tavily"] = {
            "transport": "streamable_http",
            "url": f"{TAVILY_MCP_URL}?tavilyApiKey={tavily_key}",
        }
    return connections


def hide_secrets(text: str) -> str:
    """Remove API keys from error messages before they are printed or logged."""
    for name in ("TAVILY_API_KEY", "OPENWEATHER_API_KEY", "AVIATIONSTACK_API_KEY"):
        secret = os.getenv(name)
        if secret:
            text = text.replace(secret, "***")
    return text


# =========================================================
# Running async code from normal (sync) code
# =========================================================

def run_async(coroutine):
    """Run an async call and wait for its result."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)  # normal case: no event loop running in this thread

    # Rare case (e.g. a Jupyter notebook already has a loop): use a helper thread
    outcome = {}

    def worker():
        try:
            outcome["value"] = asyncio.run(coroutine)
        except Exception as e:
            outcome["error"] = e

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


# =========================================================
# Loading the tools (once) and calling them
# =========================================================

_client = None
_tools = {}  # {"flights": {"search_flights": <tool>}, ...}
_load_errors = {}


def load_mcp_tools(force: bool = False) -> dict:
    """
    Ask every MCP server which tools it has. Done once, then remembered.
    If one server fails, the others still load (its error is kept in _load_errors).
    """
    global _client, _tools, _load_errors
    if _tools and not force:
        return _tools

    _client = MultiServerMCPClient(get_connections())

    async def load_all():
        names = list(_client.connections)
        results = await asyncio.gather(
            *(_client.get_tools(server_name=name) for name in names),
            return_exceptions=True,
        )
        return dict(zip(names, results))

    _tools, _load_errors = {}, {}
    for server, result in run_async(load_all()).items():
        if isinstance(result, BaseException):
            _load_errors[server] = hide_secrets(f"{type(result).__name__}: {result}")
            print(f"[mcp] Could not connect to '{server}' server: {_load_errors[server]}")
        else:
            _tools[server] = {tool.name: tool for tool in result}
            print(f"[mcp] Connected to '{server}': {', '.join(_tools[server])}")
    return _tools


def _to_text(result) -> str:
    """MCP tools can return plain text or a list of content blocks; turn both into text."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", block)))
        return "\n".join(p for p in parts if p)
    return str(result)


def call_mcp_tool(server: str, tool_names, arguments: dict) -> str:
    """
    Call a tool on one of the MCP servers and return its answer as text.

    tool_names can be one name or a list of possible names
    (hosted servers sometimes rename tools, e.g. "tavily-search" -> "tavily_search").

    Raises an error if the server or tool is not available, so the caller can fall back.
    """
    tools = load_mcp_tools().get(server)
    if tools is None:
        raise RuntimeError(f"MCP server '{server}' is not connected ({_load_errors.get(server, 'unknown error')})")

    if isinstance(tool_names, str):
        tool_names = [tool_names]
    tool = next((tools[name] for name in tool_names if name in tools), None)
    if tool is None:
        raise RuntimeError(f"MCP server '{server}' has no tool named {tool_names}; it has {list(tools)}")

    try:
        result = run_async(tool.ainvoke(arguments))
    except Exception as e:
        raise RuntimeError(hide_secrets(f"{tool.name} failed: {type(e).__name__}: {e}")) from None
    return _to_text(result)
