import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import new_request_state, run_travel_agent, travel_graph
from pii_filter import mask_pii
from tools.destination_tool import get_destination_preview

BASE_DIR = Path(__file__).parent

app = FastAPI(title="TripCrew AI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class PreviewRequest(BaseModel):
    q: str = ""


# =========================
# Web page
# =========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# =========================
# Travel plan (waits for all agents, then returns everything at once)
# =========================

@app.post("/api/travel")
def travel(req: TravelRequest):
    return run_travel_agent(user_input=req.message, thread_id=req.thread_id)


# =========================
# Travel plan with live progress (used by the web page)
# Runs the same graph, but sends one line of JSON for each event:
#   {"type": "privacy", "found": {"email": 1}}                          (what was masked, counts only)
#   {"type": "step", "agent": "guardrail_agent"}                        (request allowed)
#   {"type": "blocked", "category": "off_topic", "message": "..."}      (request stopped - the end)
#   {"type": "plan", "agents": ["hotel_agent", ...], "reason": "..."}   (from the supervisor)
#   {"type": "step", "agent": "hotel_agent"}                            (an agent finished)
#   {"type": "done", "answer": "...", "llm_calls": 2, "thread_id": "..."}
# =========================

@app.post("/api/travel-stream")
def travel_stream(req: TravelRequest):
    thread_id = req.thread_id or f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    start_state = new_request_state(req.message)  # personal data is masked here, before the graph

    def events():
        yield json.dumps({"type": "privacy", "found": start_state["pii_found"]}) + "\n"
        try:
            # stream_mode="updates" gives us {agent_name: what_it_returned} after each agent
            for update in travel_graph.stream(start_state, config=config, stream_mode="updates"):
                for agent, output in update.items():
                    if agent == "guardrail_agent":
                        if output and output.get("blocked"):
                            yield json.dumps({
                                "type": "blocked",
                                "category": output["block_reason"],
                                "message": output["messages"][-1].content,
                            }) + "\n"
                        else:
                            yield json.dumps({"type": "step", "agent": agent}) + "\n"
                    elif agent == "supervisor_agent":
                        # The supervisor runs between every agent; only its first run has news (the plan)
                        if output and output.get("plan"):
                            yield json.dumps({
                                "type": "plan",
                                "agents": output["plan"],
                                "reason": output.get("plan_reason", ""),
                            }) + "\n"
                    elif agent == "final_agent":
                        yield json.dumps({
                            "type": "done",
                            "answer": output["messages"][-1].content,
                            "llm_calls": output.get("llm_calls", 0),
                            "thread_id": thread_id,
                        }) + "\n"
                    else:
                        yield json.dumps({"type": "step", "agent": agent}) + "\n"
        except Exception as e:
            print(f"[travel-stream] Error: {type(e).__name__}: {e}", flush=True)
            yield json.dumps({
                "type": "error",
                "message": "Sorry, something went wrong while planning. Please try again in a minute.",
            }) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# =========================
# Destination photos for the background
#
# The web page uses POST, so the request text travels in the body and never
# appears in the server's access log (which prints every URL). The GET version
# is kept for anyone calling it directly; both mask personal data first.
# =========================

@app.post("/api/destination-preview")
def destination_preview_post(req: PreviewRequest):
    return get_destination_preview(mask_pii(req.q)[0])


@app.get("/api/destination-preview")
def destination_preview(q: str = ""):
    return get_destination_preview(mask_pii(q)[0])
