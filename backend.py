import os
import certifi
from dotenv import load_dotenv

load_dotenv(override=True)

# These help with SSL certificate verification issues when making requests to APIs
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import time
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from groq import RateLimitError

from tools.mcp_tools import call_mcp_tool

# The direct Python tools are kept as a backup, used only if an MCP server is unavailable
from tools.tavily_tool import tavily_search, format_mcp_search_result
from tools.flight_tool import search_flights
from tools.weather_planner import build_weather_plan, far_future_note, is_far_future, weather_city
from supervisor import AGENT_ORDER, decide_plan


# =========================
# API key + LLM
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=4000,        # cap on how long each answer can be (Groq counts these too)
    reasoning_effort="low",   # gpt-oss "thinks" before answering; low = fewer hidden tokens
    max_retries=0,            # retries are handled by call_llm() below
)

# Groq free tier allows 8000 tokens per minute. The prompt and the answer both count.
# Rough rule: 1 token is about 4 characters of English text.
MAX_FLIGHT_CHARS = 1200
MAX_HOTEL_CHARS = 1500
MAX_ITINERARY_CHARS = 5000


def shorten(text: str, max_chars: int) -> str:
    """Cut text down to max_chars so a big tool result can't blow up the prompt."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ...(trimmed)"


def call_llm(messages, retries: int = 4):
    """Call the LLM. If Groq says 'rate limit', wait and try again."""
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            if attempt == retries:
                raise  # give up and show the error

            # Groq tells us how long to wait in the "retry-after" header.
            # If it doesn't, wait 20s, then 40s, then 60s.
            retry_after = e.response.headers.get("retry-after")
            wait = float(retry_after) + 1 if retry_after else min(20 * attempt, 60)
            print(f"   Rate limit hit, waiting {wait:.0f}s (attempt {attempt}/{retries})...")
            time.sleep(wait)


# =========================
# State
# (the shared "notebook" every agent reads from and writes to)
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # new messages get appended
    user_query: str
    flight_results: str
    hotel_results: str
    weather_plan: str      # one labelled line per day (Outdoor / Mixed / Hot / Indoor)
    itinerary: str
    llm_calls: int
    plan: list[str]        # agents the supervisor chose, in order (final_agent always runs last)
    plan_reason: str       # the supervisor's one-sentence explanation
    completed: list[str]   # agents that have finished so far


def mark_done(state: TravelState, agent: str) -> list[str]:
    """Each agent adds its name here, so the supervisor knows who is next."""
    return state.get("completed", []) + [agent]


# =========================
# Flight Agent  (tool: flight MCP server -> AviationStack)
# =========================

def flight_agent(state: TravelState):
    print("-> Flight agent: searching flights...")

    try:
        flight_data = call_mcp_tool("flights", "search_flights", {"query": state["user_query"]})
    except Exception as e:
        print(f"   [mcp] {e}. Using the direct flight tool instead.")
        flight_data = search_flights(state["user_query"])

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight results fetched.")],
        "completed": mark_done(state, "flight_agent"),
    }


# =========================
# Hotel Agent  (tool: Tavily's hosted MCP server)
# =========================

def hotel_agent(state: TravelState):
    print("-> Hotel agent: searching hotels...")
    query = f"Best hotels for {state['user_query']}"

    try:
        raw = call_mcp_tool(
            "tavily",
            ["tavily_search", "tavily-search"],  # Tavily has used both names
            {"query": query, "max_results": 5, "search_depth": "basic"},  # basic = 1 credit
        )
        hotel_results = format_mcp_search_result(raw)  # raw JSON -> short text (fewer tokens)
    except Exception as e:
        print(f"   [mcp] {e}. Using the direct Tavily tool instead.")
        try:
            hotel_results = tavily_search(query)
        except Exception as e:
            hotel_results = f"Hotel search failed: {e}"

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched.")],
        "completed": mark_done(state, "hotel_agent"),
    }


# =========================
# Weather Agent  (tool: weather MCP server -> OpenWeather)
# No LLM: fixed rules label each day Outdoor / Mixed / Hot / Indoor (tools/weather_planner.py)
# =========================

def weather_agent(state: TravelState):
    print("-> Weather agent: checking the forecast...")
    query = state["user_query"]
    city = weather_city(query)

    if not city:
        weather_plan = "No destination found, so no weather forecast."
    elif is_far_future(query):
        weather_plan = far_future_note(city)
    else:
        try:
            forecast = call_mcp_tool("weather", "get_weather_forecast", {"city": city, "days": 5})
        except Exception as e:
            print(f"   [mcp] {e}. Using the weather server's function directly instead.")
            from mcp_servers.weather_server import get_weather_forecast
            forecast = get_weather_forecast(city, 5)
        weather_plan = build_weather_plan(query, city, forecast)

    return {
        "weather_plan": weather_plan,
        "messages": [AIMessage(content="Weather forecast checked.")],
        "completed": mark_done(state, "weather_agent"),
    }


# =========================
# Itinerary Agent  (LLM)
# =========================

def itinerary_agent(state: TravelState):
    print("-> Itinerary agent: planning the days...")
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{shorten(state.get('flight_results') or 'Not requested.', MAX_FLIGHT_CHARS)}

Hotel Results:
{shorten(state.get('hotel_results') or 'Not requested.', MAX_HOTEL_CHARS)}

Weather Plan (from the weather agent):
{state.get('weather_plan') or 'Not requested.'}

Use the weather plan: put outdoor sightseeing on Outdoor days, indoor places (museums,
markets, cafés) on Indoor days, outdoor activities early and late on Hot days, and an
indoor backup on Mixed days. Start each day with its weather label and temperature.
Make the itinerary practical, budget-aware, and easy to follow.
Keep it concise: a few bullet points per day and a short budget estimate.
If the user's budget is in Indian Rupees (₹ / INR), give all costs in INR.
"""

    response = call_llm([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt),
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "completed": mark_done(state, "itinerary_agent"),
    }


# =========================
# Final Response Agent  (LLM)
# =========================

def final_agent(state: TravelState):
    print("-> Final agent: writing the answer...")
    ran = set(state.get("completed", []))

    # Only send the data the supervisor's agents actually collected (fewer tokens),
    # and only ask for the sections that data can fill
    data_blocks, sections = [], ["Summary"]
    if "flight_agent" in ran:
        data_blocks.append(f"Flights:\n{shorten(state['flight_results'], MAX_FLIGHT_CHARS)}")
        sections.append("Flight Information")
    if "hotel_agent" in ran:
        data_blocks.append(f"Hotels:\n{shorten(state['hotel_results'], MAX_HOTEL_CHARS)}")
        sections.append("Hotel Suggestions")
    if "itinerary_agent" in ran:
        # The itinerary already contains the weather labels for each day
        data_blocks.append(f"Itinerary (written by the itinerary agent):\n{shorten(state['itinerary'], MAX_ITINERARY_CHARS)}")
        sections += ["Day-by-Day Itinerary", "Estimated Budget"]
    elif "weather_agent" in ran:
        data_blocks.append(f"Weather plan:\n{state['weather_plan']}")
        sections.append("Weather Forecast and What It Means for Your Plans")
    sections.append("Final Recommendations")

    full_trip = "itinerary_agent" in ran
    rules = [
        "Be clear and practical, and only answer what the user asked for.",
        "Clearly label all prices as rough estimates, not live quotes.",
        f"Keep the whole response under {900 if full_trip else 400} words.",
    ]
    if "flight_agent" in ran:
        rules.append("Mention that the live flight API shows flight status, not ticket prices.")
    if full_trip:
        rules.append("In the Day-by-Day Itinerary, keep each day's weather label and temperature from the itinerary.")

    final_prompt = (
        "Generate the final travel response for the user.\n\n"
        f"User Request:\n{state['user_query']}\n\n"
        + "\n\n".join(data_blocks)
        + "\n\nFormat the answer beautifully using these sections:\n"
        + "\n".join(f"{i}. {name}" for i, name in enumerate(sections, start=1))
        + "\n\nImportant:\n"
        + "\n".join(f"- {rule}" for rule in rules)
    )

    response = call_llm([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt),
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Supervisor Agent  (small LLM, once per request - see supervisor.py)
# Runs first, and again after every agent, to decide who goes next.
# =========================

def supervisor_agent(state: TravelState):
    if state.get("plan"):
        return {}  # plan already made: route_next() just picks the next agent (no LLM call)

    print("-> Supervisor: choosing which agents to run...")
    decision = decide_plan(state["user_query"])
    names = ", ".join(a.replace("_agent", "") for a in decision["agents"])
    print(f"   Plan: {names} -> final  ({decision['method']}: {decision['reason']})")

    return {
        "plan": decision["agents"],
        "plan_reason": decision["reason"],
        "completed": [],
    }


def route_next(state: TravelState) -> str:
    """The first agent in the plan that hasn't finished yet. When all are done: the final agent."""
    for agent in state.get("plan", []):
        if agent not in state.get("completed", []):
            return agent
    return "final_agent"


# =========================
# Build Graph
#
#              ┌──────────── supervisor ◄───────────┐
#   START ───► │  decides who runs next              │
#              └──► flight / hotel / weather / itinerary ──┘   (each reports back)
#                   ...and when the plan is done ──► final ──► END
# =========================

graph = StateGraph(TravelState)

graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor_agent")
graph.add_conditional_edges("supervisor_agent", route_next, AGENT_ORDER + ["final_agent"])
for agent in AGENT_ORDER:
    graph.add_edge(agent, "supervisor_agent")  # every agent reports back to the supervisor
graph.add_edge("final_agent", END)


# =========================
# Checkpointer (memory of each conversation thread)
# Uses PostgreSQL if DATABASE_URL is in .env, otherwise in-memory.
# =========================

def get_checkpointer():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print("INFO: DATABASE_URL not set -> using in-memory storage.")
        return InMemorySaver()

    # Cloud databases (Render/Neon) need SSL. Local PostgreSQL: add ?sslmode=disable
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    conn = psycopg.connect(database_url, autocommit=True, row_factory=dict_row)
    saver = PostgresSaver(conn)
    saver.setup()  # creates the checkpoint tables the first time
    print("OK: Connected to PostgreSQL.")
    return saver


checkpointer = get_checkpointer()
travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Function used by test.py and FastAPI (app.py)
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_plan": "",
            "itinerary": "",
            "llm_calls": 0,
            "plan": [],          # empty = the supervisor makes a new plan for this request
            "plan_reason": "",
            "completed": [],
        },
        config=config,
    )

    return {
        "thread_id": thread_id,
        "answer": result["messages"][-1].content,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_plan": result.get("weather_plan", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
        "plan": result.get("plan", []),
        "plan_reason": result.get("plan_reason", ""),
    }
