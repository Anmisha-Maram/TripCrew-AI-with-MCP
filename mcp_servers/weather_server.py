"""
Weather MCP server for TripCrew AI.

An MCP server is a small program that offers "tools" in a standard format
(the Model Context Protocol). Any MCP client - LangGraph, Claude Desktop,
or our test script - can discover these tools and call them.

This server has one tool: get_weather_forecast(city, days)
It uses OpenWeather's free 5-day forecast and returns one short line per day.

Run directly (the client normally starts it for you):
    python mcp_servers/weather_server.py

IMPORTANT: this server talks to its client through stdin/stdout ("stdio").
So never use print() here - it would break the conversation with the client.
Log to stderr instead (see log() below).
"""

import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import certifi
import requests
from dotenv import load_dotenv

# Harmless warning from a library the mcp package uses (Python 3.14); hide it to keep logs clean
warnings.filterwarnings("ignore", message=".*has an incomplete definition.*")

from mcp.server.fastmcp import FastMCP

# The client may start this file from any folder, so load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# These help with SSL certificate verification issues when making requests to APIs
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
MAX_DAYS = 5  # the free forecast covers about 5 days

# log_level="WARNING" hides the "Processing request of type ..." line for every call
mcp = FastMCP("weather", log_level="WARNING")


def log(message: str):
    """Write to stderr, because stdout is used for talking to the MCP client."""
    print(f"[weather-server] {message}", file=sys.stderr)


# =========================================================
# Helpers
# =========================================================

def summarize_day(readings: list) -> dict:
    """Turns the 3-hourly readings of one day into a short summary."""
    temps_min = [r["main"]["temp_min"] for r in readings]
    temps_max = [r["main"]["temp_max"] for r in readings]

    # Describe the day using the daytime readings (6am-9pm) if there are any
    daytime = [r for r in readings if 6 <= r["local_hour"] <= 21] or readings
    descriptions = Counter(r["weather"][0]["description"] for r in daytime)
    condition = descriptions.most_common(1)[0][0]

    # Mention rain/snow even if it's only part of the day ("clear sky, light rain at times").
    # OpenWeather condition ids below 700 are rain, drizzle, snow or storms.
    wet = Counter(r["weather"][0]["description"] for r in daytime if r["weather"][0]["id"] < 700)
    is_wet = any(word in condition for word in ("rain", "drizzle", "snow", "thunder", "shower"))
    if wet and not is_wet:
        condition += f", {wet.most_common(1)[0][0]} at times"

    # Storms matter for planning even if they only show up once
    if any(200 <= r["weather"][0]["id"] < 300 for r in readings) and "thunder" not in condition:
        condition += ", thunderstorms possible"

    rain_chance = max(r.get("pop", 0) for r in readings)              # 0.0 - 1.0
    rain_mm = sum((r.get("rain") or {}).get("3h", 0) for r in readings)
    wind_kmh = max(r["wind"]["speed"] for r in readings) * 3.6         # m/s -> km/h

    return {
        "min": round(min(temps_min)),
        "max": round(max(temps_max)),
        "condition": condition,
        "rain_chance": round(rain_chance * 100),
        "rain_mm": round(rain_mm, 1),
        "wind_kmh": round(wind_kmh),
        "partial": len(readings) < 4,  # e.g. today, when only the evening is left
    }


def format_day(day, summary: dict) -> str:
    """Thu 2 Oct: 24-31°C, light rain, 70% chance of rain (3.2 mm), wind up to 12 km/h"""
    line = (
        f"{day.strftime('%a')} {day.day} {day.strftime('%b')}: "
        f"{summary['min']}-{summary['max']}°C, {summary['condition']}, "
        f"{summary['rain_chance']}% chance of rain"
    )
    if summary["rain_mm"] > 0:
        line += f" ({summary['rain_mm']} mm)"
    line += f", wind up to {summary['wind_kmh']} km/h"
    if summary["partial"]:
        line += " (rest of today only)"
    return line


# =========================================================
# The MCP tool
# =========================================================

@mcp.tool()
def get_weather_forecast(city: str, days: int = 5) -> str:
    """
    Get the daily weather forecast for a city, for up to 5 days from today.

    Args:
        city: City name, optionally with a country code, e.g. "Jaipur" or "Paris, FR".
        days: Number of days to return (1-5). Defaults to 5.

    Returns one short line per day: temperature range, conditions,
    chance of rain and wind.
    """
    if not API_KEY:
        return "Weather error: OPENWEATHER_API_KEY is missing. Add it to your .env file."

    city = (city or "").strip()
    if not city:
        return "Weather error: please give a city name."

    requested_days = int(days)
    days = max(1, min(requested_days, MAX_DAYS))

    try:
        response = requests.get(
            FORECAST_URL,
            params={"q": city, "appid": API_KEY, "units": "metric"},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        log(f"request failed: {e}")
        return f"Weather error: could not reach OpenWeather ({type(e).__name__})."

    if response.status_code == 401:
        return ("Weather error: the OpenWeather key was rejected. New keys can take "
                "up to 2 hours to start working; otherwise check OPENWEATHER_API_KEY.")
    if response.status_code == 404:
        return f"Weather error: city '{city}' was not found. Try adding a country code, e.g. 'Jaipur, IN'."
    if response.status_code == 429:
        return "Weather error: too many requests to OpenWeather. Please wait a minute."
    if response.status_code != 200:
        return f"Weather error: OpenWeather returned status {response.status_code}."

    try:
        data = response.json()
        offset = data["city"]["timezone"]  # seconds from UTC, e.g. 19800 for India

        # Group the 3-hourly readings by the city's LOCAL date
        by_day = defaultdict(list)
        for reading in data["list"]:
            local = datetime.fromtimestamp(reading["dt"] + offset, tz=timezone.utc)
            reading["local_hour"] = local.hour
            by_day[local.date()].append(reading)

        all_days = sorted(by_day)
        chosen = all_days[:days]
        lines = [format_day(day, summarize_day(by_day[day])) for day in chosen]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        log(f"unexpected response: {e}")
        return "Weather error: OpenWeather sent an unexpected response."

    place = f"{data['city'].get('name', city)}, {data['city'].get('country', '')}".strip(", ")
    result = f"Weather forecast for {place} (local time):\n" + "\n".join(lines)

    if requested_days > len(chosen):
        result += (f"\nNote: you asked for {requested_days} days, but only {len(chosen)} are "
                   "available (the free forecast covers about 5 days).")

    log(f"forecast sent for {place}, {len(chosen)} days")
    return result


if __name__ == "__main__":
    mcp.run()  # default transport is "stdio"
