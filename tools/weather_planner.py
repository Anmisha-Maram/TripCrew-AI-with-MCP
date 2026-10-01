"""
Helpers for the weather agent (no LLM, no API calls here):

    weather_city(query)     "5 days in Jaipur"      -> "Jaipur, IN"
    trip_length(query)      "a week in Goa"         -> 7
    is_far_future(query)    "Paris in December"     -> True (no 5-day forecast possible)
    build_weather_plan(...) forecast text           -> one labelled line per trip day

Day labels (simple fixed rules, so it's always clear why a day got its label):
    🌧️ Indoor day   rain chance >= 60%, heavy rain (>= 5 mm) or thunderstorms
    🥵 Hot day      38°C or hotter
    🌦️ Mixed day    rain chance 30-59%
    ☀️ Outdoor day  everything else

Try it:  python tools/weather_planner.py
"""

import re
import sys
from datetime import date
from pathlib import Path

# So this file also works when run on its own (python tools/weather_planner.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pycountry

from tools.flight_tool import (
    AIRPORTS,
    CITY_MAIN_AIRPORT,
    COUNTRY_ALIASES,
    COUNTRY_MAIN_AIRPORT,
    detect_destination,
    resolve_location_to_iata,
)

DEFAULT_TRIP_DAYS = 3
FORECAST_DAYS = 5          # the free OpenWeather forecast covers about 5 days

# Rules for the day labels (change these numbers to tune them)
INDOOR_RAIN_CHANCE = 60    # %
HEAVY_RAIN_MM = 5
MIXED_RAIN_CHANCE = 30     # %
HOT_DAY_C = 38

LABELS = {
    "indoor": ("🌧️ Indoor day", "plan museums, markets, cafés and other indoor places"),
    "hot": ("🥵 Hot day", "do outdoor sights early morning and evening, stay indoors at midday"),
    "mixed": ("🌦️ Mixed day", "outdoor sights in the morning, keep an indoor backup"),
    "outdoor": ("☀️ Outdoor day", "good for sightseeing, walks and viewpoints"),
}


# =========================================================
# Reading the trip request
# =========================================================

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14,
}


def trip_length(query: str):
    """How many days the trip lasts, or None if the request doesn't say."""
    q = query.lower()
    for word, number in NUMBER_WORDS.items():
        q = re.sub(rf"\b{word}\b", str(number), q)

    match = re.search(r"\b(\d+)\s*-?\s*(day|days|night|nights)\b", q)
    if match:
        days = int(match.group(1))
        if match.group(2).startswith("night"):
            days += 1  # 4 nights = 5 days
        return max(1, min(days, 30))

    match = re.search(r"\b(\d+)\s*-?\s*weeks?\b", q)
    if match:
        return max(1, min(int(match.group(1)) * 7, 30))
    if re.search(r"\b(a|one)\s+week\b|\bweek-long\b|\bweeklong\b", q):
        return 7
    if "fortnight" in q:
        return 14
    if "weekend" in q:
        return 2
    return None


MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def is_far_future(query: str, today: date | None = None) -> bool:
    """
    True if the trip is clearly later than the 5-day forecast can see,
    e.g. "in December", "next month", "in 2027".
    """
    today = today or date.today()
    q = query.lower()

    if re.search(r"\bnext\s+(month|year)\b|\bin\s+\d+\s+(weeks|months)\b", q):
        return True
    if any(int(y) > today.year for y in re.findall(r"\b(20\d\d)\b", q)):
        return True

    for number, month in enumerate(MONTHS, start=1):
        if month == "may":
            # "may" is also a normal word ("I may travel"), so only count it as a month here
            found = re.search(r"\b(in|of|during|this|next|early|late|mid)\s+may\b|\d+(st|nd|rd|th)?\s+may\b|\bmay\s+\d", q)
        else:
            found = re.search(rf"\b{month}\b|\b{month[:3]}\b(?=\s*\d)|\d\s*{month[:3]}\b", q)
        if found and number != today.month:
            return True
    return False


def _country_code(name: str):
    name = name.lower()
    if name in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[name]
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        return None


def weather_city(query: str):
    """
    Which city to get the forecast for, with a country code so OpenWeather
    picks the right place ("Goa" alone could match a town in another country).
    For a whole country we use its main city: "India" -> "Delhi, IN".
    """
    destination = detect_destination(query)
    if not destination:
        return None

    # A country? Use the city of its main airport.
    # (Known cities like Bali, Dubai and Singapore are also in COUNTRY_ALIASES, so check cities first.)
    code = None if destination.lower() in CITY_MAIN_AIRPORT else _country_code(destination)
    if code and code in COUNTRY_MAIN_AIRPORT:
        airport = AIRPORTS.get(COUNTRY_MAIN_AIRPORT[code], {})
        city = airport.get("city")
        if city:
            return f"{city}, {code}"

    # A city: add the country of its airport
    iata = resolve_location_to_iata(destination)
    country = AIRPORTS.get(iata, {}).get("country") if iata else None
    return f"{destination}, {country}" if country else destination


# =========================================================
# Turning the forecast into labelled trip days
# =========================================================

# Matches the lines from the weather MCP server, e.g.
# "Fri 2 Oct: 24-35°C, clear sky, light rain at times, 40% chance of rain (3.2 mm), wind up to 18 km/h"
DAY_LINE = re.compile(
    r"^(?P<date>\w{3} \d{1,2} \w{3}): (?P<min>-?\d+)-(?P<max>-?\d+)°C, (?P<condition>.+?), "
    r"(?P<chance>\d+)% chance of rain(?: \((?P<mm>[\d.]+) mm\))?, wind up to (?P<wind>\d+) km/h"
    r"(?P<partial> \(rest of today only\))?"
)


def parse_forecast(forecast_text: str) -> list[dict]:
    """Reads the weather server's answer back into numbers. Skips 'rest of today' lines."""
    days = []
    for line in forecast_text.splitlines():
        match = DAY_LINE.match(line.strip())
        if not match or match.group("partial"):
            continue  # not a forecast line, or only the evening of today is left
        days.append({
            "date": match.group("date"),
            "min": int(match.group("min")),
            "max": int(match.group("max")),
            "condition": match.group("condition"),
            "rain_chance": int(match.group("chance")),
            "rain_mm": float(match.group("mm") or 0),
        })
    return days


def label_day(day: dict) -> str:
    """Apply the rules from the top of this file."""
    if ("thunder" in day["condition"] or day["rain_chance"] >= INDOOR_RAIN_CHANCE
            or day["rain_mm"] >= HEAVY_RAIN_MM):
        return "indoor"
    if day["max"] >= HOT_DAY_C:
        return "hot"
    if day["rain_chance"] >= MIXED_RAIN_CHANCE:
        return "mixed"
    return "outdoor"


def build_weather_plan(query: str, city: str, forecast_text: str) -> str:
    """
    The short text the itinerary agent receives, e.g.

    Weather plan for Goa, IN (live forecast, trip assumed to start tomorrow):
    Day 1 (Fri 2 Oct): ☀️ Outdoor day, 26-33°C, broken clouds, 21% rain. Good for sightseeing...
    Day 2 (Sat 3 Oct): 🌧️ Indoor day, 24-35°C, overcast clouds, 100% rain. Plan museums...
    """
    days = parse_forecast(forecast_text)
    if not days:
        # The weather server sent an error message (or something unexpected)
        reason = forecast_text.strip().splitlines()[0].rstrip(".") if forecast_text.strip() else "no data"
        return f"No live forecast for {city} ({reason}). Plan for the typical weather there."

    trip_days = trip_length(query) or DEFAULT_TRIP_DAYS
    lines = [f"Weather plan for {city} (live forecast, trip assumed to start tomorrow):"]

    for number, day in enumerate(days[:trip_days], start=1):
        label, advice = LABELS[label_day(day)]
        lines.append(
            f"Day {number} ({day['date']}): {label}, {day['min']}-{day['max']}°C, "
            f"{day['condition']}, {day['rain_chance']}% rain. {advice.capitalize()}."
        )

    if trip_days > len(days):
        first_missing = len(days) + 1
        span = f"Day {first_missing}" if first_missing == trip_days else f"Days {first_missing}-{trip_days}"
        lines.append(f"{span}: no forecast yet (the free forecast covers about {FORECAST_DAYS} days). "
                     "Keep these days flexible.")
    return "\n".join(lines)


def far_future_note(city: str) -> str:
    return (f"The trip is later than the 5-day forecast can see, so there is no live forecast for {city}. "
            "Plan for the typical weather there in that season.")


# =========================================================
# Demo: python tools/weather_planner.py  (no API calls)
# =========================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("Reading trip requests:")
    for q in ["Plan a 5-day trip to Jaipur", "a week in Goa from Mumbai", "weekend in Paris",
              "4 nights in Tokyo", "plan an itinerary from usa to india", "Paris in December",
              "I may travel to Rome soon", "trip to Bali next month", "Plan a trip from Delhi to JFK"]:
        print(f"  {q!r:42} city={weather_city(q)!s:20} days={trip_length(q)!s:5} far_future={is_far_future(q)}")

    sample = """Weather forecast for Goa, IN (local time):
Thu 1 Oct: 27-30°C, clear sky, 0% chance of rain, wind up to 9 km/h (rest of today only)
Fri 2 Oct: 26-33°C, broken clouds, 21% chance of rain (0.4 mm), wind up to 10 km/h
Sat 3 Oct: 24-35°C, overcast clouds, light rain at times, 100% chance of rain (4.0 mm), wind up to 13 km/h
Sun 4 Oct: 25-34°C, scattered clouds, light rain at times, 45% chance of rain (1.1 mm), wind up to 12 km/h
Mon 5 Oct: 27-39°C, clear sky, 0% chance of rain, wind up to 8 km/h
Tue 6 Oct: 25-31°C, thunderstorm, 55% chance of rain (2.0 mm), wind up to 20 km/h"""

    print("\nSample forecast -> weather plan for a 7-day trip:")
    print(build_weather_plan("a 7 day trip to Goa", "Goa, IN", sample))

    print("\nWhen the weather server sends an error:")
    print(build_weather_plan("3 days in Goa", "Goa, IN", "Weather error: could not reach OpenWeather (Timeout)."))
