import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.flight_tool import detect_destination

load_dotenv(override=True)

# A small, cheap model just for the attraction list.
# On Groq each model has its own per-minute token limit, so these calls
# do not eat into the budget the 4 travel agents (gpt-oss-120b) need.
preview_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=300,
    reasoning_effort="low",
    max_retries=1,
    model_kwargs={"response_format": {"type": "json_object"}},  # JSON only
)

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikipedia asks every app to say who it is
WIKI_HEADERS = {"User-Agent": "TripCrewAI/1.0 (student travel-planner project)"}

NUM_ATTRACTIONS = 5
# Big enough for a full-screen background. Wikimedia only serves standard widths
# (e.g. 960, 1280, 1920); others like 1600 return an error.
IMAGE_WIDTH = 1280

# Shown before the user types a destination (or if something fails)
DEFAULT_PLACES = [
    ("Santorini", "Whitewashed villages above the Aegean Sea"),
    ("Machu Picchu", "Inca citadel high in the Andes"),
    ("Taj Mahal", "Marble mausoleum on the Yamuna river"),
    ("Banff National Park", "Turquoise lakes in the Canadian Rockies"),
    ("Kyoto", "Temples, gardens and old wooden streets"),
]

# Memory cache: {"jaipur": {...result...}}. Cleared when the server restarts.
_cache = {}
_cache_lock = threading.Lock()


# =========================================================
# Wikipedia photos
# =========================================================

def get_wikipedia_photo(title: str):
    """Returns (image_url, article_url, short_description, width, height) for a Wikipedia article, or None."""
    try:
        response = requests.get(
            WIKI_SUMMARY_URL + quote(title.replace(" ", "_")),
            headers=WIKI_HEADERS,
            timeout=8,
        )
        if response.status_code != 200:
            return None
        data = response.json()
    except (requests.exceptions.RequestException, ValueError):
        return None

    original = data.get("originalimage") or {}
    thumbnail = data.get("thumbnail") or {}
    if not original.get("source"):
        return None

    # Original photos can be huge (10+ MB). If it is wider than we need,
    # ask Wikipedia for a resized copy by changing the width in the thumbnail URL.
    image = original["source"]
    if original.get("width", 0) > IMAGE_WIDTH and thumbnail.get("source"):
        image = re.sub(r"/\d+px-", f"/{IMAGE_WIDTH}px-", thumbnail["source"])

    article = ((data.get("content_urls") or {}).get("desktop") or {}).get("page", "")
    # Width and height let the page tell tall photos from wide ones
    return image, article, data.get("description", ""), original.get("width", 0), original.get("height", 0)


def add_photos(places):
    """places: list of dicts with name / wiki / caption. Keeps only places that have a photo."""
    def fetch(place):
        photo = get_wikipedia_photo(place.get("wiki") or place["name"])
        if not photo and place.get("wiki"):
            photo = get_wikipedia_photo(place["name"])  # try the plain name too
        if not photo:
            return None
        image, article, description, width, height = photo
        return {
            "name": place["name"],
            "image": image,
            "caption": place.get("caption") or description,
            "source": article,
            "width": width,
            "height": height,
        }

    # Fetch all photos at the same time instead of one after another
    with ThreadPoolExecutor(max_workers=NUM_ATTRACTIONS) as pool:
        results = [r for r in pool.map(fetch, places) if r]

    # Wide photos first: they fill a desktop screen best, so the first impression looks great
    results.sort(key=lambda r: r["width"] < r["height"])
    return results


# =========================================================
# Attractions from Groq
# =========================================================

def ask_groq_for_attractions(destination: str):
    """One short Groq call. Returns a list of {"name", "wiki", "caption"}."""
    response = preview_llm.invoke([
        SystemMessage(content="You are a travel guide. Reply with JSON only."),
        HumanMessage(content=(
            f"List the {NUM_ATTRACTIONS} most famous tourist attractions in {destination}. "
            'Format: {"attractions": [{"name": "...", "wiki": "exact English Wikipedia '
            'article title", "caption": "max 8 words"}]}'
        )),
    ])

    # Pull the JSON object out of the reply, even if extra text sneaks in
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if not match:
        return []
    data = json.loads(match.group(0))

    places = []
    for item in data.get("attractions", [])[:NUM_ATTRACTIONS]:
        if isinstance(item, dict) and item.get("name"):
            places.append({
                "name": str(item["name"]),
                "wiki": str(item.get("wiki") or ""),
                "caption": str(item.get("caption") or ""),
            })
    return places


# =========================================================
# Main function used by app.py
# =========================================================

def get_default_preview():
    """General travel photos. Uses Wikipedia only (no Groq), and is cached."""
    with _cache_lock:
        if "__default__" in _cache:
            return _cache["__default__"]

    places = [{"name": name, "caption": caption} for name, caption in DEFAULT_PLACES]
    result = {"destination": None, "attractions": add_photos(places), "is_default": True}

    if result["attractions"]:  # only cache if Wikipedia actually worked
        with _cache_lock:
            _cache["__default__"] = result
    return result


def get_destination_preview(text: str):
    """
    'Plan a 5-day trip to Jaipur' ->
    {"destination": "Jaipur", "attractions": [{"name", "image", "caption", "source"}, ...]}
    Never raises: on any problem it returns the default travel background.
    """
    try:
        destination = detect_destination(text or "")
        if not destination:
            return get_default_preview()

        key = destination.lower()
        with _cache_lock:
            if key in _cache:
                return _cache[key]  # already looked up: no Groq call

        print(f"[preview] Looking up attractions for {destination}...")
        attractions = add_photos(ask_groq_for_attractions(destination))
        if not attractions:
            return get_default_preview()

        result = {"destination": destination, "attractions": attractions, "is_default": False}
        with _cache_lock:
            _cache[key] = result
        return result

    except Exception as e:  # rate limit, bad JSON, network error...
        print(f"[preview] Failed, using default background: {e}")
        return get_default_preview()
