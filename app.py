import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend import run_travel_agent, travel_graph
from tools.destination_tool import get_destination_preview

BASE_DIR = Path(__file__).parent

app = FastAPI(title="TripCrew AI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


# =========================
# Web page
# =========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# =========================
# Travel plan (waits for all 4 agents, then returns everything at once)
# =========================

@app.post("/api/travel")
def travel(req: TravelRequest):
    return run_travel_agent(user_input=req.message, thread_id=req.thread_id)


# =========================
# Travel plan with live progress (used by the web page)
# Runs the same graph, but sends one line of JSON each time an agent finishes:
#   {"type": "step", "agent": "hotel_agent"}
#   {"type": "done", "answer": "...", "llm_calls": 2, "thread_id": "..."}
# =========================

@app.post("/api/travel-stream")
def travel_stream(req: TravelRequest):
    thread_id = req.thread_id or f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    start_state = {
        "messages": [HumanMessage(content=req.message)],
        "user_query": req.message,
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "llm_calls": 0,
    }

    def events():
        try:
            # stream_mode="updates" gives us {agent_name: what_it_returned} after each agent
            for update in travel_graph.stream(start_state, config=config, stream_mode="updates"):
                for agent, output in update.items():
                    if agent == "final_agent":
                        yield json.dumps({
                            "type": "done",
                            "answer": output["messages"][-1].content,
                            "llm_calls": output.get("llm_calls", 0),
                            "thread_id": thread_id,
                        }) + "\n"
                    else:
                        yield json.dumps({"type": "step", "agent": agent}) + "\n"
        except Exception as e:
            print(f"[travel-stream] Error: {e}")
            yield json.dumps({
                "type": "error",
                "message": "Sorry, something went wrong while planning. Please try again in a minute.",
            }) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# =========================
# Destination photos for the background
# =========================

@app.get("/api/destination-preview")
def destination_preview(q: str = ""):
    return get_destination_preview(q)
