# TripCrew AI ✈️

**A multi-agent AI travel planner built with LangGraph.**
Describe your trip in plain English, like *"plan a 5 day trip from Hyderabad to Delhi"*, and a crew of four AI agents finds live flights, searches hotels, plans each day and writes a complete travel plan. While they work, the page shows photos of the destination's top attractions.

![TripCrew AI planning a trip to Paris](docs/screenshots/paris-planning.png)

---

## ✨ Features

- **4-agent LangGraph pipeline:** Flight → Hotel → Itinerary → Final agents share one state and each adds its part.
- **Live flight data** from AviationStack, with a custom parser that turns free text like *"Nepal trip from India"* into airport codes (DEL → KTM).
- **Hotel and destination research** using Tavily web search.
- **Destination-aware gallery:** the app detects the city you type and shows a slideshow of its top attractions, with photos and captions from Wikipedia.
- **Live agent progress:** each agent's status (*Working… → Done*) streams to the page as it runs.
- **Download your plan as a PDF**, with a clean, printer-friendly layout.
- **Conversation memory** with LangGraph's PostgreSQL checkpointer, falling back to in-memory storage when no database is set.
- **Free-tier friendly:** trimmed tool results, capped response length, low reasoning effort and automatic retry when Groq's rate limit is hit.

## 📸 Screenshots

| Delhi: Red Fort | Delhi: India Gate |
|---|---|
| ![All four agents done for a Delhi trip](docs/screenshots/delhi-red-fort.png) | ![India Gate in the attraction gallery](docs/screenshots/delhi-india-gate.png) |

---

## 🧠 How it works

```
User request
    │
    ▼
┌──────────────┐   ┌──────────────┐   ┌────────────────┐   ┌──────────────┐
│ Flight agent │ → │ Hotel agent  │ → │ Itinerary agent│ → │ Final agent  │
│ AviationStack│   │ Tavily search│   │ Groq LLM       │   │ Groq LLM     │
└──────────────┘   └──────────────┘   └────────────────┘   └──────────────┘
                                                                  │
                         State saved after every step             ▼
                         (PostgreSQL checkpointer)          Final travel plan
```

| Agent | What it does | Tool |
|---|---|---|
| **Flight agent** | Works out the route from your text and fetches live flights | AviationStack API |
| **Hotel agent** | Searches the web for hotels at the destination | Tavily |
| **Itinerary agent** | Writes a budget-aware, day-by-day plan | Groq (`openai/gpt-oss-120b`) |
| **Final agent** | Turns everything into a polished plan: summary, flights, hotels, itinerary, budget and tips | Groq (`openai/gpt-oss-120b`) |

The **attraction gallery** runs separately. A small model (`openai/gpt-oss-20b`) suggests the destination's top attractions, Wikipedia supplies the photos, and results are cached per city so the same destination doesn't use tokens twice.

## 🛠️ Tech stack

| Area | Tools |
|---|---|
| Agents and orchestration | LangGraph, LangChain |
| LLM | Groq: `openai/gpt-oss-120b` (agents), `openai/gpt-oss-20b` (attractions) |
| Tools and APIs | AviationStack, Tavily, Wikipedia REST API |
| Backend | Python 3.11, FastAPI, Uvicorn, streaming responses |
| Database | PostgreSQL (LangGraph checkpointer) |
| Frontend | HTML, CSS, JavaScript, marked + DOMPurify (Markdown), html2pdf.js (PDF) |
| Location data | pycountry, airportsdata |

## 📁 Project structure

```
TripCrew-AI/
├── app.py                  # FastAPI server: web page + API endpoints
├── backend.py              # LangGraph graph: the 4 agents, state and checkpointer
├── test.py                 # Run the planner from the terminal
├── requirements.txt
├── tools/
│   ├── flight_tool.py      # AviationStack + text → airport-code route parsing
│   ├── tavily_tool.py      # Hotel / web search
│   └── destination_tool.py # Attractions + Wikipedia photos for the gallery
├── templates/
│   └── index.html          # Web page
└── static/
    ├── style.css
    └── script.js           # Gallery, live progress, PDF download
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

### 2. Clone the repository

```bash
git clone https://github.com/Anmisha-Maram/TripCrew-AI.git
cd TripCrew-AI
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
uvicorn app:app --reload
```

Open **<http://127.0.0.1:8000>** in your browser and try:

- *plan a 5 day trip from Hyderabad to Delhi*
- *plan a trip from Delhi to Paris*
- *7 day trip from Mumbai to Dubai under ₹1,50,000*

### 7. (Optional) Run in the terminal

```bash
python test.py
```

---

## 🔌 API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `POST` | `/api/travel` | Runs all agents and returns the full plan. Body: `{"message": "...", "thread_id": "optional"}` |
| `POST` | `/api/travel-stream` | Same as above, but streams each agent's progress as JSON lines (used by the web page) |
| `GET` | `/api/destination-preview?q=Paris` | Top attractions and photos for a destination |

## ⚠️ Notes and limitations

- **No ticket prices:** AviationStack provides live flight *status*, not fares. Prices in the plan are AI estimates and are labelled as such.
- **Groq free tier:** each model has a tokens-per-minute limit. The app trims tool results and retries automatically, but very large requests may take a little longer.
- **AviationStack free plan:** 100 requests per month.
- Attraction photos come from Wikipedia and are credited under each image.

## 🗺️ Roadmap

- [ ] Weather agent with a custom **MCP** server (OpenWeather)
- [ ] Tools connected through the **Model Context Protocol (MCP)**
- [ ] **Supervisor agent** that decides which agents to run
- [ ] **Input guardrails** to block off-topic requests
- [ ] **Human-in-the-loop:** approve or revise the plan before it's finalised
- [ ] Deploy online

## 🙏 Acknowledgements

Inspired by [entbappy's TripMate AI](https://github.com/entbappy/TripMate-AI-A-Multi-Agent-Travel-Planner-with-LangGraph) tutorial. Built on top of it: the destination-aware attraction gallery, live agent progress streaming, PDF export, the migration to AviationStack's APILayer API, and token optimisation for Groq's free tier.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Author:** [Anmisha Maram](https://github.com/Anmisha-Maram)
