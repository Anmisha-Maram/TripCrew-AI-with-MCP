"""
The supervisor's "brain": decides which agents a request needs.

    decide_plan("Find good hotels in Goa")
    -> {"agents": ["hotel_agent"], "reason": "User only wants hotels", "method": "llm"}

How it decides:
  1. One small LLM call (openai/gpt-oss-20b, ~300 tokens, its own Groq rate limit).
  2. Safety rules in code (always applied):
       - only known agents, always in the fixed order flight -> hotel -> weather -> itinerary
       - flights only if the request says where you travel FROM, or asks about flights
       - an empty plan becomes a full trip plan
     (The final agent is not in the plan: it always runs last.)
  3. If the LLM call fails, simple keyword rules decide instead.

Try it:
    python supervisor.py          # keyword rules only (free, no API calls)
    python supervisor.py --llm    # the real LLM supervisor (~300 tokens per request)
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.flight_tool import resolve_location_to_iata

load_dotenv(override=True)

# The order agents always run in (data first, then the plan that uses it)
AGENT_ORDER = ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"]

# A small, cheap model. On Groq each model has its own per-minute token limit,
# so the supervisor does not use the budget of the main agents (gpt-oss-120b).
supervisor_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=300,
    reasoning_effort="low",
    max_retries=1,
    model_kwargs={"response_format": {"type": "json_object"}},  # JSON only
)

SUPERVISOR_PROMPT = """You are the supervisor of a travel-planning team. Choose which agents the user's request needs.

Agents:
- flight: live flights. ONLY if the user says where they travel from, or asks about flights.
- hotel: hotels and places to stay.
- weather: the weather forecast.
- itinerary: a day-by-day trip plan.

A full trip plan needs hotel, weather and itinerary (plus flight if a starting city is given).
A narrow question needs only its agent, e.g. "hotels in Goa" -> hotel.

Reply with JSON only: {"agents": ["..."], "reason": "one short sentence"}"""


# =========================================================
# Safety rules
# =========================================================

def mentions_origin_or_flights(query: str) -> bool:
    """Does the request say where the trip starts, or ask about flights?"""
    q = query.lower()
    if re.search(r"\b(flight|flights|fly|flying|airline|airlines|airfare|plane)\b", q):
        return True
    # "from Delhi", "from New York to ..." - only if it's really a place
    for match in re.finditer(r"\bfrom\s+([a-z][a-z .]+?)(?=\s+(?:to|for|on|in|with|under|next)\b|[,.!?]|$)", q):
        if resolve_location_to_iata(match.group(1).strip()):
            return True
    # Two airport codes: "DEL to GOI"
    return bool(re.search(r"\b[A-Z]{3}\s+to\s+[A-Z]{3}\b", query))


def to_agent_name(name: str):
    """'flights', 'Flight', 'flight_agent' -> 'flight_agent'. Unknown names -> None."""
    name = str(name).lower().strip()
    for agent in AGENT_ORDER:
        if name.startswith(agent.split("_")[0]):
            return agent
    return None


def clean_plan(agents, query: str) -> list[str]:
    """Apply the safety rules to whatever the LLM (or the keywords) chose."""
    chosen = {to_agent_name(a) for a in agents} - {None}

    if "flight_agent" in chosen and not mentions_origin_or_flights(query):
        chosen.discard("flight_agent")  # don't show default-origin flights nobody asked for

    if not chosen:  # nothing usable: treat it as a full trip plan
        chosen = {"hotel_agent", "weather_agent", "itinerary_agent"}
        if mentions_origin_or_flights(query):
            chosen.add("flight_agent")

    return [agent for agent in AGENT_ORDER if agent in chosen]  # fixed order


# =========================================================
# Keyword rules (backup when the LLM is not available)
# =========================================================

KEYWORDS = {
    "hotel_agent": r"\b(hotel|hotels|stay|stays|accommodation|resort|resorts|hostel|airbnb|lodging)\b",
    "weather_agent": r"\b(weather|forecast|rain|raining|temperature|climate|sunny|hot|cold)\b",
    "itinerary_agent": r"\b(plan|planning|itinerary|trip|visit|tour|vacation|holiday|"
                       r"honeymoon|getaway|things to do|sightseeing)\b",
}

# Words like "3 days" or "weekend" suggest a trip, but "weather this weekend" is not a trip plan,
# so these only count when nothing more specific matched
WEAK_TRIP_WORDS = r"\b(days?|nights?|week|weekend)\b"


def keyword_plan(query: str) -> tuple[list[str], str]:
    q = query.lower()
    chosen = [agent for agent, pattern in KEYWORDS.items() if re.search(pattern, q)]
    if not chosen and re.search(WEAK_TRIP_WORDS, q):
        chosen = ["itinerary_agent"]

    # A trip plan also needs hotels and weather
    if "itinerary_agent" in chosen:
        chosen += ["hotel_agent", "weather_agent"]
    if mentions_origin_or_flights(query):
        chosen.append("flight_agent")

    plan = clean_plan(chosen, query)
    return plan, "Chosen by keyword rules: " + ", ".join(a.replace("_agent", "") for a in plan)


# =========================================================
# Main function used by backend.py
# =========================================================

def decide_plan(query: str) -> dict:
    """Which agents to run for this request (the final agent always runs last)."""
    try:
        response = supervisor_llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"User request: {query}"),
        ])
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}

        agents = data.get("agents", [])
        if not isinstance(agents, list) or not agents:
            raise ValueError(f"no agents in reply: {response.content[:100]}")

        plan = clean_plan(agents, query)
        reason = str(data.get("reason") or "").strip() or "Chosen by the supervisor."
        return {"agents": plan, "reason": reason, "method": "llm"}

    except Exception as e:  # rate limit, bad JSON, network...
        print(f"   [supervisor] LLM decision failed ({type(e).__name__}); using keyword rules.")
        plan, reason = keyword_plan(query)
        return {"agents": plan, "reason": reason, "method": "keywords"}


# =========================================================
# Demo
# =========================================================

EXAMPLES = [
    "Plan a 3 day trip from Delhi to Goa",
    "Find good hotels in Goa",
    "What's the weather in Paris this weekend?",
    "Flights from Mumbai to Dubai",
    "Plan 4 days in Jaipur",
    "Where should I stay in Tokyo, and will it rain?",
    "Plan a trip to Bali",
    "hello",
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    use_llm = "--llm" in sys.argv

    print("Supervisor decisions using", "the LLM (gpt-oss-20b)" if use_llm else "keyword rules only (free)")
    for query in EXAMPLES:
        if use_llm:
            result = decide_plan(query)
            agents, reason = result["agents"], f"[{result['method']}] {result['reason']}"
        else:
            agents, reason = keyword_plan(query)
        names = " -> ".join(a.replace("_agent", "") for a in agents + ["final_agent"])
        print(f"\n  {query!r}\n    runs:   {names}\n    reason: {reason}")
