import os
import re

import certifi
import airportsdata
import pycountry
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# These help with SSL certificate verification issues when making requests to APIs
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Accepts either spelling of the key name in .env
API_KEY = os.getenv("AVIATIONSTACK_API_KEY") or os.getenv("AVIATION_STACK_API_KEY")

# Used when the user only mentions a destination, e.g. "Japan trip"
DEFAULT_ORIGIN_IATA = os.getenv("DEFAULT_ORIGIN_IATA", "DEL")  # Delhi

# The free AviationStack plan only supports http (not https)
BASE_URL = "https://api.apilayer.net/aviationstack/v1/flights"

AIRPORTS = airportsdata.load("IATA")


# =========================================================
# Lookup tables (all keys lowercase, because clean_text() lowercases)
# =========================================================

# Everyday names that pycountry does not recognise on its own
COUNTRY_ALIASES = {
    "usa": "US",
    "america": "US",
    "united states": "US",
    "united states of america": "US",
    "uk": "GB",
    "britain": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "uae": "AE",
    "dubai": "AE",
    "south korea": "KR",
    "korea": "KR",
    "russia": "RU",
    "holland": "NL",
    "turkey": "TR",
    "vietnam": "VN",
    "bali": "ID",
    "maldives": "MV",
    "india": "IN",
    "nepal": "NP",
    "sri lanka": "LK",
    "bangladesh": "BD",
    "bhutan": "BT",
    "japan": "JP",
    "china": "CN",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "indonesia": "ID",
    "qatar": "QA",
    "saudi arabia": "SA",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
    "switzerland": "CH",
    "netherlands": "NL",
}

# Main airport for each country (keys are 2-letter country CODES)
COUNTRY_MAIN_AIRPORT = {
    "IN": "DEL",
    "US": "JFK",
    "GB": "LHR",
    "AE": "DXB",
    "JP": "NRT",
    "CN": "PEK",
    "KR": "ICN",
    "SG": "SIN",
    "MY": "KUL",
    "TH": "BKK",
    "ID": "CGK",
    "NP": "KTM",
    "LK": "CMB",
    "BD": "DAC",
    "BT": "PBH",
    "MV": "MLE",
    "VN": "SGN",
    "QA": "DOH",
    "SA": "JED",
    "TR": "IST",
    "CA": "YYZ",
    "AU": "SYD",
    "DE": "FRA",
    "FR": "CDG",
    "IT": "FCO",
    "ES": "MAD",
    "NL": "AMS",
    "CH": "ZRH",
    "RU": "SVO",
    "BR": "GRU",
    "MX": "MEX",
}

# Main airport for popular cities
CITY_MAIN_AIRPORT = {
    # India
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "bombay": "BOM",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "kolkata": "CCU",
    "goa": "GOI",
    "bali": "DPS",
    "phuket": "HKT",
    "jaipur": "JAI",
    "kochi": "COK",
    "ahmedabad": "AMD",
    "pune": "PNQ",
    # World
    "new york": "JFK",
    "london": "LHR",
    "paris": "CDG",
    "berlin": "BER",
    "frankfurt": "FRA",
    "rome": "FCO",
    "madrid": "MAD",
    "amsterdam": "AMS",
    "zurich": "ZRH",
    "tokyo": "NRT",
    "osaka": "KIX",
    "seoul": "ICN",
    "beijing": "PEK",
    "shanghai": "PVG",
    "singapore": "SIN",
    "bangkok": "BKK",
    "kuala lumpur": "KUL",
    "dubai": "DXB",
    "doha": "DOH",
    "istanbul": "IST",
    "kathmandu": "KTM",
    "colombo": "CMB",
    "dhaka": "DAC",
    "male": "MLE",
    "toronto": "YYZ",
    "sydney": "SYD",
    "moscow": "SVO",
    "sao paulo": "GRU",
    "mexico city": "MEX",
}


# =========================================================
# Text helpers
# =========================================================

def clean_text(text: str) -> str:
    """Lowercase, remove symbols and drop travel filler words."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    stop_words = [
        "flights", "flight", "airline", "airlines", "airport", "airports",
        "plane", "planes", "ticket", "tickets", "travel", "travelling",
        "trip", "trips", "journey", "journeys", "plan", "complete",
        "day", "days", "including", "info", "information", "hotel",
        "hotels", "sightseeing", "under", "budget",
    ]
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words).strip()


def country_name_to_code(text: str):
    """'india' -> 'IN', 'trip to dubai' -> 'AE'. Returns None if not found."""
    text = clean_text(text)

    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]

    try:
        country = pycountry.countries.lookup(text)
        return country.alpha_2
    except LookupError:
        pass

    # Detect a country name inside longer text
    for country in pycountry.countries:
        country_name = country.name.lower()
        if len(country_name) >= 4 and re.search(rf"\b{re.escape(country_name)}\b", text):
            return country.alpha_2

    for alias, code in COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return code

    return None


# =========================================================
# Airport helpers
# =========================================================

def airport_country_matches(airport: dict, country_code: str) -> bool:
    airport_country = str(airport.get("country", "")).upper().strip()
    return airport_country == country_code


def get_best_airport_for_country(country_code: str):
    """Pick the main airport for a country code, e.g. 'IN' -> 'DEL'."""
    preferred = COUNTRY_MAIN_AIRPORT.get(country_code)

    if preferred and preferred in AIRPORTS:
        return preferred

    candidates = []

    for iata, airport in AIRPORTS.items():
        if not iata:
            continue

        if airport_country_matches(airport, country_code):
            name = str(airport.get("name", "")).lower()
            city = str(airport.get("city", "")).lower()

            score = 0
            if "international" in name:
                score += 50
            if "intl" in name:
                score += 40
            if "capital" in name:
                score += 20
            if city:
                score += 5

            candidates.append((score, iata))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_location_to_iata(location: str):
    """
    Converts a country / city / IATA code into an airport IATA code.
    Examples: 'India' -> DEL, 'Goa' -> GOI, 'Tokyo' -> NRT, 'BOM' -> BOM
    """
    if not location:
        return None

    raw_location = location.strip()
    location_clean = clean_text(raw_location)

    if not location_clean:
        return None

    # City preferred airport (checked first so "Goa" isn't read as code GOA = Genoa)
    if location_clean in CITY_MAIN_AIRPORT:
        return CITY_MAIN_AIRPORT[location_clean]

    # Direct IATA code
    if re.fullmatch(r"[A-Za-z]{3}", raw_location):
        code = raw_location.upper()
        if code in AIRPORTS:
            return code

    # Country preferred airport
    country_code = country_name_to_code(location_clean)
    if country_code:
        airport = get_best_airport_for_country(country_code)
        if airport:
            return airport

    # City match from the airport database
    city_matches = []

    for iata, airport in AIRPORTS.items():
        city = str(airport.get("city", "")).lower().strip()
        name = str(airport.get("name", "")).lower().strip()

        score = 0
        if city == location_clean:
            score += 100
        elif location_clean in city:
            score += 70
        if location_clean in name:
            score += 50
        if "international" in name:
            score += 10

        if score > 0:
            city_matches.append((score, iata))

    if city_matches:
        city_matches.sort(reverse=True)
        return city_matches[0][1]

    return None


# =========================================================
# Query parsing
# =========================================================

def find_location_mentions(query: str):
    """Finds country or city names inside a natural-language query."""
    q = query.lower()
    mentions = []

    for alias in COUNTRY_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", q):
            mentions.append(alias)

    for country in pycountry.countries:
        name = country.name.lower()
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", q):
            mentions.append(name)

    for city in CITY_MAIN_AIRPORT:
        if re.search(rf"\b{re.escape(city)}\b", q):
            mentions.append(city)

    # Remove duplicates while keeping order
    unique_mentions = []
    for item in mentions:
        if item not in unique_mentions:
            unique_mentions.append(item)

    return unique_mentions


# Words that usually end the place name, e.g. "to Goa FOR 4 days"
_END = r"(?:\s+(?:on|for|under|including|with|in|at|next|this|during)\b|[.,!?]|$)"


def parse_route(query: str):
    """
    Returns (dep_iata, arr_iata). Either can be None:
      None, None -> global live flights
      DEL, GOI   -> route
      DEL, None  -> all flights from DEL
      None, GOI  -> all flights to GOI
    """
    q = query.strip()
    q_lower = q.lower()

    global_keywords = [
        "all country", "all countries", "global flight", "global flights",
        "all flight", "all flights", "worldwide flight", "worldwide flights",
    ]
    if any(keyword in q_lower for keyword in global_keywords):
        return None, None

    # Direct IATA codes, e.g. "DEL to GOI"
    codes = [c for c in re.findall(r"\b[A-Z]{3}\b", q) if c in AIRPORTS]
    if len(codes) >= 2:
        return codes[0], codes[1]

    # "from X to Y"
    match = re.search(rf"\bfrom\s+(.+?)\s+to\s+(.+?){_END}", q_lower)
    if match:
        return (
            resolve_location_to_iata(match.group(1)),
            resolve_location_to_iata(match.group(2)),
        )

    # "to Y from X"
    match = re.search(rf"\bto\s+(.+?)\s+from\s+(.+?){_END}", q_lower)
    if match:
        return (
            resolve_location_to_iata(match.group(2)),
            resolve_location_to_iata(match.group(1)),
        )

    # "Nepal trip from India" / "flights from X"
    match = re.search(rf"\bfrom\s+(.+?){_END}", q_lower)
    if match:
        dep_iata = resolve_location_to_iata(match.group(1))
        # Look for a destination in the text before "from"
        before = q_lower[: match.start()]
        mentions = find_location_mentions(before)
        arr_iata = resolve_location_to_iata(mentions[0]) if mentions else None
        return dep_iata, arr_iata

    # "flights to X"
    match = re.search(rf"\bto\s+(.+?){_END}", q_lower)
    if match:
        return DEFAULT_ORIGIN_IATA, resolve_location_to_iata(match.group(1))

    # Fallback: look for any city/country names
    mentions = find_location_mentions(q)

    if len(mentions) >= 2:
        return (
            resolve_location_to_iata(mentions[0]),
            resolve_location_to_iata(mentions[1]),
        )

    if len(mentions) == 1:
        return DEFAULT_ORIGIN_IATA, resolve_location_to_iata(mentions[0])

    return None, None


# =========================================================
# Destination name (used by the frontend background)
# =========================================================

# "jaipur" -> "Jaipur", built once from the airport database
_AIRPORT_CITIES = {}
for _airport in AIRPORTS.values():
    _city = str(_airport.get("city", "")).strip()
    if _city:
        _AIRPORT_CITIES.setdefault(_city.lower(), _city)


def _country_display_name(country_code: str) -> str:
    """'IN' -> 'India', 'KR' -> 'South Korea'."""
    country = pycountry.countries.get(alpha_2=country_code)
    if not country:
        return country_code
    return getattr(country, "common_name", None) or country.name


def _place_name(text: str):
    """Turns a short piece of text like 'jaipur' or 'india' into a nice place name."""
    cleaned = clean_text(text)
    if not cleaned:
        return None

    # 1. Popular city we already know, e.g. "goa", "new york"
    if cleaned in CITY_MAIN_AIRPORT:
        return cleaned.title()

    # 2. Exact country, e.g. "india", "usa"
    if cleaned in COUNTRY_ALIASES:
        return _country_display_name(COUNTRY_ALIASES[cleaned])
    try:
        return _country_display_name(pycountry.countries.lookup(cleaned).alpha_2)
    except LookupError:
        pass

    # 3. Any city in the airport database, e.g. "udaipur"
    if cleaned in _AIRPORT_CITIES:
        return _AIRPORT_CITIES[cleaned]

    # 4. An airport code, e.g. "jfk" -> "New York"
    if re.fullmatch(r"[a-z]{3}", cleaned) and cleaned.upper() in AIRPORTS:
        city = AIRPORTS[cleaned.upper()].get("city")
        if city:
            return city

    # 5. A known city or country somewhere inside the text (the earliest one wins)
    mentions = find_location_mentions(cleaned)
    if mentions:
        first = min(mentions, key=lambda m: re.search(rf"\b{re.escape(m)}\b", cleaned).start())
        return _place_name(first) if first != cleaned else first.title()

    return None


def detect_destination(query: str):
    """
    Finds where the user wants to GO (not where they leave from).
    Examples: 'Plan a 5-day trip to Jaipur' -> 'Jaipur'
              'from USA to India for 7 days' -> 'India'
              'Japan trip for 7 days'        -> 'Japan'
    Returns None if no place is found.
    """
    q_lower = query.strip().lower()
    if not q_lower:
        return None

    # Two airport codes, e.g. "LHR to CDG" (same rule as parse_route): the second one is the destination
    codes = [c for c in re.findall(r"\b[A-Z]{3}\b", query) if c in AIRPORTS]
    if len(codes) >= 2 and AIRPORTS[codes[1]].get("city"):
        return AIRPORTS[codes[1]]["city"]

    candidates = []

    # "to Y" (and drop any "from X" that comes after it)
    match = re.search(rf"\bto\s+(.+?){_END}", q_lower)
    if match:
        candidates.append(re.split(r"\bfrom\b", match.group(1))[0])

    # "Japan trip from India" -> the part before "from"
    match = re.search(r"^(.*?)\bfrom\b", q_lower)
    if match:
        candidates.append(match.group(1))

    # Otherwise, the whole text, minus any "from X" part (that's the origin, not the destination)
    candidates.append(re.sub(r"\bfrom\s+.+?(?=\bto\b|$)", " ", q_lower))

    for text in candidates:
        name = _place_name(text)
        if name:
            return name

    return None


# =========================================================
# Formatting + API call
# =========================================================

def format_flight(flight: dict) -> str:
    """One short line per flight. Every word here is sent to the LLM, so keep it small."""
    airline = (flight.get("airline") or {}).get("name") or "Unknown airline"
    flight_number = (flight.get("flight") or {}).get("iata") or "?"
    status = flight.get("flight_status") or "unknown"

    dep = flight.get("departure") or {}
    arr = flight.get("arrival") or {}

    # "2026-10-01T13:00:00+00:00" -> "2026-10-01 13:00"
    dep_time = (dep.get("scheduled") or "?")[:16].replace("T", " ")
    arr_time = (arr.get("scheduled") or "?")[:16].replace("T", " ")

    delay = dep.get("delay")
    delay_text = f", delay {delay} min" if delay else ""

    return (
        f"- {flight_number} ({airline}): {dep.get('iata') or '?'} {dep_time} -> "
        f"{arr.get('iata') or '?'} {arr_time}, {status}{delay_text}"
    )


def search_flights(query: str, limit: int = 3) -> str:
    """Main function used by the flight agent."""
    if not API_KEY:
        return (
            "Flight API error: AVIATIONSTACK_API_KEY is missing.\n"
            "Add this line to your .env file:\n"
            "AVIATIONSTACK_API_KEY=your_api_key_here"
        )

    dep_iata, arr_iata = parse_route(query)

    params = {
        "access_key": API_KEY,
        "limit": min(limit, 100),
    }
    if dep_iata:
        params["dep_iata"] = dep_iata
    if arr_iata:
        params["arr_iata"] = arr_iata

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Flight API request failed: {e}"
    except ValueError:
        return "Flight API returned invalid JSON."

    if "error" in data:
        error = data["error"]
        return (
            "Flight API error:\n"
            f"Code: {error.get('code', 'Unknown')}\n"
            f"Message: {error.get('message', 'Unknown error')}"
        )

    flight_data = data.get("data", [])

    if dep_iata and arr_iata:
        route_text = f"from {dep_iata} to {arr_iata}"
    elif dep_iata:
        route_text = f"from {dep_iata}"
    elif arr_iata:
        route_text = f"to {arr_iata}"
    else:
        route_text = "worldwide"

    if not flight_data:
        return (
            f"No live flight data found {route_text}.\n\n"
            "Note: AviationStack provides live flight status, not ticket prices. "
            "For fares, check sites like Google Flights or MakeMyTrip."
        )

    formatted_flights = [format_flight(f) for f in flight_data[:limit]]

    return f"Live flights {route_text}:\n" + "\n".join(formatted_flights)


if __name__ == "__main__":
    print(parse_route("Plan a 4 days trip from Delhi to Goa in December"))
    print(parse_route("Japan trip for 7 days"))
    print(parse_route("flights from Mumbai to Dubai"))
    print("\n" + "=" * 60 + "\n")
    print(search_flights("Plan a 4 days trip from Delhi to Goa"))