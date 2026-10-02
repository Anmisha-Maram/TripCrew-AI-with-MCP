# TripCrew AI with MCP ✈️

**A multi-agent AI travel planner built with LangGraph and the Model Context Protocol (MCP).**

🌐 **Live demo: <https://tripcrew-ai-with-mcp.onrender.com>** (hosted on Render's free plan, so the first visit may take up to a minute to wake up)

Describe your trip in plain English, like *"plan a 5 day trip from Hyderabad to Delhi"*. A **supervisor agent** decides which agents your request needs, and a crew of AI agents finds live flights, searches hotels, checks the weather, plans each day and writes a complete travel plan. Before anything runs, a **privacy filter** hides your personal data and a **guardrail** stops unsafe requests. At the end, **you review the plan**: approve it, or ask for changes. While the agents work, the page shows photos of the destination's top attractions.

> ✅ **Phase 2 complete.** This project builds on [TripCrew AI](https://github.com/Anmisha-Maram/TripCrew-AI) (the LangGraph-only version). Phase 2 moved the tools to **MCP servers**, added a **weather agent**, a **supervisor agent**, **guardrails**, a **PII filter** and **human-in-the-loop** approval, and deployed the app on **Render**. See **Phase 2 progress** below.

---

## 🌐 Try it online

Open **<https://tripcrew-ai-with-mcp.onrender.com>** in any browser, on a computer or phone. No sign-up or installation is needed.

1. **Wait for it to wake up.** The app runs on Render's free plan, which puts it to sleep after about 15 minutes without visitors. The first visit after that can take up to a minute; after that it's fast.
2. **Describe your trip** in the box, for example:
   - *Plan a 3 day trip from Delhi to Goa* (full trip: flights, hotels, weather, itinerary)
   - *Find good hotels in Jaipur* (the supervisor runs only the hotel agent)
   - *What's the weather in Goa this week?* (weather only)
3. **Watch the crew work.** The sidebar shows each step live: privacy filter, guardrail, supervisor, then only the agents your request needs.
4. **Review the draft.** Click **✓ Approve plan**, or type what should change (e.g. *"make day 2 more relaxed"*) and click **✎ Request changes** (up to 3 rounds).
5. **Download the PDF.** The **⬇ Download PDF** button appears once you approve the plan.

> Good to know: the demo shares free-tier API limits with every visitor (Groq tokens per minute, 100 AviationStack flight requests per month), so a plan can take a little longer when it's busy. Personal data such as emails and phone numbers is masked before it reaches the AI or the database, but it's still best not to enter real personal details in a demo.

## ✨ Features

- **Supervisor + 5 agents (LangGraph):** the supervisor reads your request and runs only the agents it needs (e.g. hotels only, weather only, or the full trip). Every agent reports back to the supervisor, which picks the next one.
- **Tools on MCP servers:** weather and flights run as custom MCP servers, and hotel search uses Tavily's hosted MCP server, all connected through `MultiServerMCPClient`. If a server is down, the agents automatically use the direct Python tools instead.
- **Weather agent:** labels each trip day 🌧️ Indoor, 🥵 Hot, 🌦️ Mixed or ☀️ Outdoor from the 5-day forecast, using fixed rules (0 tokens). The itinerary puts museums on rainy days and sightseeing on sunny ones.
- **Live flight data** from AviationStack, with a custom parser that turns free text like *"Nepal trip from India"* into airport codes (DEL → KTM).
- **Input guardrails:** off-topic, unsafe and prompt-injection requests are stopped before any agent or tool runs. Three layers, cheapest first: code rules → Llama Prompt Guard 2 (0 tokens) → a small LLM only for unclear requests.
- **PII filter:** emails, phone numbers, Aadhaar, PAN, passport and card numbers are masked (`[EMAIL]`, `[PHONE]`, …) before they reach the LLMs, the logs or the database. Checksums (Luhn, Verhoeff) keep false alarms low.
- **Human-in-the-loop review:** the plan pauses as a *draft* (LangGraph `interrupt()`). Approve it, or type what should change and the crew rewrites it (up to 3 rounds). **Only an approved plan can be downloaded as a PDF.** Change requests also go through the PII filter and guardrail.
- **Destination-aware gallery:** the app detects the city you type and shows a slideshow of its top attractions, with photos and captions from Wikipedia.
- **Live agent progress:** each step's status (*Working… → Done*) streams to the page as it runs, including which agents the supervisor skipped and why.
- **Download your plan as a PDF**, with a clean, printer-friendly layout.
- **Conversation memory** with LangGraph's PostgreSQL checkpointer (through a connection pool that replaces dropped connections), falling back to in-memory storage when no database is set.
- **Free-tier friendly:** trimmed tool results, capped response length, low reasoning effort, small models for the supervisor and guardrail, and automatic retry when Groq's rate limit is hit.

---

## 🧠 How it works

```
Your request
    │
    ▼
🔒 Privacy filter ──► 🛡️ Guardrail ──blocked──► friendly message (END)
                          │ allowed
                          ▼
              ┌──── 🧭 Supervisor ◄──────────────────────────────┐
              │   picks the agents                                │ each agent
              └──► ✈️ Flight / 🏨 Hotel / ☀️ Weather / 🗺️ Itinerary ┘ reports back
                          │ plan done
                          ▼
                   📝 Final agent ──► 🙋 Your review ──approve──► ✅ Approved plan + PDF
                          ▲                 │
                          └──── changes ────┘  (up to 3 rounds)

            State is saved after every step (PostgreSQL checkpointer),
            so a plan can wait for your review as long as needed.
```

| Step | What it does | Tool / model |
|---|---|---|
| **Privacy filter** | Masks personal data before the graph starts, so nothing after it sees the real values | Regex + checksums (no AI) |
| **Guardrail** | Stops off-topic, unsafe and prompt-injection requests | Code rules → Llama Prompt Guard 2 → `openai/gpt-oss-20b` (only if unclear) |
| **Supervisor** | Decides which agents run, in a fixed order. Flights only run if you say where you travel from or ask about flights | `openai/gpt-oss-20b` (one small call), keyword rules as backup |
| **Flight agent** | Works out the route from your text and fetches live flights | Flight MCP server → AviationStack API |
| **Hotel agent** | Searches the web for hotels at the destination | Tavily's hosted MCP server |
| **Weather agent** | Labels each day Indoor / Hot / Mixed / Outdoor | Weather MCP server → OpenWeather, fixed rules (0 tokens) |
| **Itinerary agent** | Writes a budget-aware, day-by-day plan that follows the weather labels | Groq (`openai/gpt-oss-120b`) |
| **Final agent** | Turns everything into a polished plan: summary, flights, hotels, itinerary, budget and tips | Groq (`openai/gpt-oss-120b`) |
| **Your review** | Pauses the graph until you approve or ask for changes | LangGraph `interrupt()` / `Command(resume=...)` |

The **attraction gallery** runs separately. A small model (`openai/gpt-oss-20b`) suggests the destination's top attractions, Wikipedia supplies the photos, and results are cached per city so the same destination doesn't use tokens twice.

## 🛠️ Tech stack

| Area | Tools |
|---|---|
| Agents and orchestration | LangGraph (supervisor graph, `interrupt()` for human review), LangChain |
| LLM | Groq: `openai/gpt-oss-120b` (itinerary + final agents), `openai/gpt-oss-20b` (supervisor, guardrail, attractions) |
| Safety | Llama Prompt Guard 2 (`meta-llama/llama-prompt-guard-2-86m`), PII filter with Luhn / Verhoeff checksums |
| Tools and APIs | AviationStack, Tavily, OpenWeather, Wikipedia REST API |
| MCP | `mcp` (FastMCP servers), `langchain-mcp-adapters` |
| Backend | Python 3.11, FastAPI, Uvicorn, streaming responses |
| Database | PostgreSQL (LangGraph checkpointer, `psycopg_pool` connection pool) |
| Frontend | HTML, CSS, JavaScript, marked + DOMPurify (Markdown), html2pdf.js (PDF) |
| Location data | pycountry, airportsdata |

## 📁 Project structure

```
TripCrew-AI-with-MCP/
├── app.py                  # FastAPI server: web page, live progress stream, review endpoint
├── backend.py              # LangGraph graph: agents, human review, state and checkpointer
├── supervisor.py           # Supervisor: chooses which agents run for each request
├── guardrails.py           # Input guardrails: rules, Prompt Guard 2, small LLM topic check
├── pii_filter.py           # Masks personal data before the LLMs, logs or database
├── test.py                 # Run the planner from the terminal (with review)
├── requirements.txt
├── render.yaml             # Render deployment blueprint (no keys inside)
├── .python-version         # Python 3.11, used by Render
├── tools/
│   ├── flight_tool.py      # AviationStack + text → airport-code route parsing
│   ├── tavily_tool.py      # Hotel / web search
│   ├── weather_planner.py  # Indoor / Hot / Mixed / Outdoor day labels (no LLM)
│   ├── destination_tool.py # Attractions + Wikipedia photos for the gallery
│   └── mcp_tools.py        # Connects to all MCP servers (MultiServerMCPClient)
├── mcp_servers/
│   ├── weather_server.py       # Weather MCP server (OpenWeather 5-day forecast)
│   ├── flight_server.py        # Flight MCP server (wraps tools/flight_tool.py)
│   ├── test_weather_server.py  # Tests the weather server as a real MCP client
│   └── test_mcp_tools.py       # Tests all 3 MCP servers together
├── templates/
│   └── index.html          # Web page
└── static/
    ├── style.css
    └── script.js           # Gallery, live progress, review panel, PDF download
```

---

## 🚀 Getting started

### 1. Prerequisites

- **Python 3.11** (Anaconda/Miniconda recommended)
- **Git**
- **PostgreSQL** (optional: without it, the app keeps plans in memory)
- Free API keys:
  - Groq: <https://console.groq.com/keys>
  - Tavily: <https://app.tavily.com>
  - AviationStack: <https://aviationstack.com> (sign up on the APILayer dashboard)
  - OpenWeather: <https://home.openweathermap.org/api_keys> (new keys can take up to 2 hours to activate)

### 2. Clone the repository

```bash
git clone https://github.com/Anmisha-Maram/TripCrew-AI-with-MCP.git
cd TripCrew-AI-with-MCP
```

### 3. Create and activate an environment

**Option A: conda**
```bash
conda create -n travel python=3.11 -y
conda activate travel
```

**Option B: venv**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create your `.env` file

Create a file named **`.env`** in the project's root folder, next to `app.py`:

```env
# Groq (LLM)
GROQ_API_KEY=your_groq_api_key

# Tavily (hotel / web search)
TAVILY_API_KEY=your_tavily_api_key

# AviationStack (live flights)
AVIATIONSTACK_API_KEY=your_aviationstack_api_key

# OpenWeather (weather MCP server)
OPENWEATHER_API_KEY=your_openweather_api_key

# Optional: airport used when only a destination is given (default: DEL)
DEFAULT_ORIGIN_IATA=DEL

# Optional: PostgreSQL for saving conversation state
# Local PostgreSQL:
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/postgres?sslmode=disable
# Cloud PostgreSQL (Render / Neon): paste the external URL, SSL is added automatically
```

> 🔒 `.env` is listed in `.gitignore`, so your keys are never committed. Never share it or upload it.
> If you skip `DATABASE_URL`, the app runs with in-memory storage.

### 6. Run the web app

```bash
python -m uvicorn app:app --reload
```

Open **<http://127.0.0.1:8000>** in your browser and try:

- *plan a 5 day trip from Hyderabad to Delhi* (full trip: all agents)
- *find good hotels in Jaipur* (the supervisor runs only the hotel agent)
- *what's the weather in Goa this week?* (weather only)
- *7 day trip from Mumbai to Dubai under ₹1,50,000*

When the draft appears, approve it to unlock **Download PDF**, or type a change such as *"make day 2 more relaxed"* and click **Request changes**.

### 7. (Optional) Run in the terminal

```bash
python test.py
```

After the draft is printed, press Enter to approve it or type what should change.

### 8. (Optional) Test the parts on their own

These demos are free (no LLM tokens) unless a flag says otherwise:

```bash
python pii_filter.py              # masks sample emails, phones, Aadhaar, PAN, passport, cards
python guardrails.py              # ~15 sample requests through the code rules
python guardrails.py --all        # the same requests through all 3 layers (uses Groq)
python supervisor.py              # which agents each sample request gets (keyword rules)
python supervisor.py --llm        # the real LLM supervisor (~300 tokens per request)
python tools/weather_planner.py   # day labels from sample forecasts
```

### 9. (Optional) Test the MCP servers

```bash
# Weather server on its own: lists its tools and asks for forecasts for a few cities
python mcp_servers/test_weather_server.py

# All 3 servers (weather, flights, Tavily) through MultiServerMCPClient, one call each
python mcp_servers/test_mcp_tools.py
```

---

## ☁️ Deploy on Render

The live demo runs on [Render](https://render.com) with this setup:

| Setting | Value |
|---|---|
| Service type | Web Service (free instance), auto-deploys from the `main` branch |
| Region | Oregon (same region as the database) |
| Runtime | Python 3.11 (from `.python-version`) |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Database | Render PostgreSQL, connected through its **Internal Database URL** |
| Secrets | API keys and `DATABASE_URL` set as environment variables in the Render dashboard (never in the code) |

The weather and flight MCP servers start inside the same service (as `stdio` subprocesses), and Tavily's MCP server is hosted by Tavily, so no extra services are needed.

**To deploy your own copy:**

1. Create a **PostgreSQL** database on Render (recommended: plans waiting for your review are saved there and survive restarts; without it, the app uses in-memory storage).
2. Click **New → Blueprint** and connect your fork of this repository; Render reads the included **`render.yaml`**. (Or click **New → Web Service** and enter the build and start commands above yourself.)
3. Add the environment variables: `GROQ_API_KEY`, `TAVILY_API_KEY`, `AVIATIONSTACK_API_KEY`, `OPENWEATHER_API_KEY` and `DATABASE_URL` (the database's **Internal Database URL**; the web service must be in the same region).
4. Deploy. The logs should show `OK: Connected to PostgreSQL.` and `Your service is live 🎉`, then open your `.onrender.com` link.

> On the free plan the service sleeps after about 15 minutes without visitors, so the first request after that takes up to a minute. Render's free PostgreSQL databases also expire after a while, so check the expiry date on the database page.

## 🔌 API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `POST` | `/api/travel` | Runs the agents and returns the draft plan. Body: `{"message": "...", "thread_id": "optional"}`. The reply's `status` is `awaiting_review`, `approved` or `blocked` |
| `POST` | `/api/travel-stream` | Same as above, but streams progress as JSON lines (used by the web page): `privacy`, `step`, `plan`, `blocked`, `review` (draft ready, graph paused) and `error` |
| `POST` | `/api/travel-review` | Answers a paused draft. Body: `{"thread_id": "...", "action": "approve" \| "change", "feedback": "what to change"}`. Streams the same events, ending with a new `review` (after changes) or `done` (approved) |
| `POST` | `/api/destination-preview` | Top attractions and photos for a destination. Body: `{"q": "Paris"}` (a `GET` version with `?q=` also works) |

## ⚠️ Notes and limitations

- **No ticket prices:** AviationStack provides live flight *status*, not fares. Prices in the plan are AI estimates and are labelled as such.
- **Weather:** the forecast covers the next 5 days only. For trips further away, the plan says so instead of guessing.
- **PII filter:** names and street addresses are not masked (they have no fixed pattern).
- **Groq free tier:** each model has a tokens-per-minute limit. The app trims tool results and retries automatically, but very large requests may take a little longer. Approving a plan costs 0 tokens; each change round costs 1–2 LLM calls.
- **AviationStack free plan:** 100 requests per month.
- Attraction photos come from Wikipedia and are credited under each image.

## 🗺️ Phase 2 progress

- [x] **Step 1: Weather MCP server.** A custom OpenWeather server (`mcp_servers/weather_server.py`), standalone and tested
- [x] **Step 2: Tools on MCP.** Weather MCP, a flight MCP server wrapping the AviationStack tool, and Tavily's hosted MCP, connected to LangGraph with `langchain-mcp-adapters` (`MultiServerMCPClient`), with automatic fallback to the direct tools if a server is unavailable
- [x] **Step 3: Weather agent.** Labels each day Indoor / Hot / Mixed / Outdoor from the forecast with fixed rules (`tools/weather_planner.py`, 0 tokens), and the itinerary follows the labels
- [x] **Step 4: Supervisor agent.** One small LLM call reads the request and decides which agents run (e.g. hotels only, weather only, or the full trip); agents report back to the supervisor after each step (`supervisor.py`)
- [x] **Step 5: Input guardrails.** Off-topic, unsafe and prompt-injection requests are blocked before any tools run: code rules → Llama Prompt Guard 2 → small LLM only for unclear requests (`guardrails.py`)
- [x] **Step 6: PII filter.** Emails, phone numbers, passport, Aadhaar, PAN and card numbers are masked before they reach the LLMs, logs or database (`pii_filter.py`)
- [x] **Step 7: Human-in-the-loop.** The graph pauses after the final answer with LangGraph `interrupt()`; you approve or request changes in the UI (up to 3 rounds), and only approved plans can be downloaded as PDF
- [x] **Deploy online.** Live on Render at <https://tripcrew-ai-with-mcp.onrender.com> (web service + Render PostgreSQL, `render.yaml` blueprint)

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Author:** [Anmisha Maram](https://github.com/Anmisha-Maram)
