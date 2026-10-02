import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend import (
    new_request_state,
    review_resume,
    run_travel_agent,
    travel_graph,
    waiting_for_review,
)
from pii_filter import mask_pii
from tools.destination_tool import get_destination_preview

BASE_DIR = Path(__file__).parent

app = FastAPI(title="TripCrew AI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ReviewRequest(BaseModel):
    thread_id: str
    action: str            # "approve" or "change"
    feedback: str = ""     # what to change (only for "change")


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
#   {"type": "review", "answer": "...", "can_change": true, ...}        (draft ready: graph paused)
# =========================

def json_line(event: dict) -> str:
    return json.dumps(event) + "\n"


def stream_graph(graph_input, thread_id: str):
    """Runs (or resumes) the graph and turns each update into one event line."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # stream_mode="updates" gives us {agent_name: what_it_returned} after each agent
        for update in travel_graph.stream(graph_input, config=config, stream_mode="updates"):
            for agent, output in update.items():
                if agent == "__interrupt__":
                    # The graph paused in human_review: the draft waits for the user
                    draft = output[0].value
                    values = travel_graph.get_state(config).values
                    yield json_line({
                        "type": "review",
                        "answer": draft["answer"],
                        "revisions": draft["revisions"],
                        "can_change": draft["can_change"],
                        "llm_calls": values.get("llm_calls", 0),
                        "thread_id": thread_id,
                    })
                elif agent == "human_review":
                    if output and output.get("approved"):
                        values = travel_graph.get_state(config).values
                        yield json_line({
                            "type": "done",
                            "answer": values["messages"][-1].content,
                            "llm_calls": values.get("llm_calls", 0),
                            "thread_id": thread_id,
                        })
                elif agent == "guardrail_agent":
                    if output and output.get("blocked"):
                        yield json_line({
                            "type": "blocked",
                            "category": output["block_reason"],
                            "message": output["messages"][-1].content,
                        })
                    else:
                        yield json_line({"type": "step", "agent": agent})
                elif agent == "supervisor_agent":
                    # The supervisor runs between every agent; only its first run has news (the plan)
                    if output and output.get("plan"):
                        yield json_line({
                            "type": "plan",
                            "agents": output["plan"],
                            "reason": output.get("plan_reason", ""),
                        })
                else:
                    yield json_line({"type": "step", "agent": agent})
    except Exception as e:
        print(f"[travel-stream] Error: {type(e).__name__}: {e}", flush=True)
        yield json_line({
            "type": "error",
            "message": "Sorry, something went wrong while planning. Please try again in a minute.",
        })


@app.post("/api/travel-stream")
def travel_stream(req: TravelRequest):
    thread_id = req.thread_id or f"user_{uuid.uuid4().hex}"
    start_state = new_request_state(req.message)  # personal data is masked here, before the graph

    def events():
        yield json_line({"type": "privacy", "found": start_state["pii_found"]})
        yield from stream_graph(start_state, thread_id)

    return StreamingResponse(events(), media_type="application/x-ndjson")


# =========================
# Human review: approve the draft, or ask for changes (resumes the paused graph)
# Sends the same events as above, plus:
#   {"type": "privacy", "found": {...}}                (what was masked in the change request)
#   {"type": "review_blocked", "message": "..."}       (change request refused; still paused)
#   {"type": "done", "answer": "...", ...}             (approved: the plan can be downloaded)
# =========================

@app.post("/api/travel-review")
def travel_review(req: ReviewRequest):
    def events():
        draft = waiting_for_review(req.thread_id)
        if not draft:
            yield json_line({
                "type": "error",
                "message": "This plan is no longer waiting for review. Please plan the trip again.",
            })
            return
        if req.action == "change" and not draft["can_change"]:
            yield json_line({"type": "review_blocked", "message": "No more changes allowed. Please approve the plan."})
            return

        resume = review_resume(req.action, req.feedback)  # masks and checks the change request
        if not resume["ok"]:
            yield json_line({"type": "review_blocked", "message": resume["message"]})
            return

        yield json_line({"type": "privacy", "found": resume["pii_found"]})
        yield from stream_graph(resume["command"], req.thread_id)

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
