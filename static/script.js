// =========================================================
// Settings
// =========================================================

const TYPING_PAUSE_MS = 800;     // wait this long after typing stops before looking up the destination
const SLIDE_EVERY_MS = 5000;     // change background photo every 5 seconds

// The order the 5 agents run in (same as backend.py)
const AGENTS = ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent", "final_agent"];

// Four slow zoom/pan styles (see style.css), used in turn
const KEN_BURNS = ["kb-1", "kb-2", "kb-3", "kb-4"];

// =========================================================
// Page elements
// =========================================================

const form = document.getElementById("trip-form");
const input = document.getElementById("trip-input");
const button = document.getElementById("plan-button");
const progress = document.getElementById("progress");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");
const resultBody = document.getElementById("result-body");
const resultMeta = document.getElementById("result-meta");
const downloadButton = document.getElementById("download-button");
const chip = document.getElementById("destination-chip");
const caption = document.getElementById("caption");
const captionDots = document.getElementById("caption-dots");
const captionName = document.getElementById("caption-name");
const captionText = document.getElementById("caption-text");
const captionCredit = document.getElementById("caption-credit");
const layers = [document.getElementById("bg-a"), document.getElementById("bg-b")];

// Lets the CSS progress bars use the same timing as the slideshow
document.documentElement.style.setProperty("--slide-ms", `${SLIDE_EVERY_MS}ms`);

// =========================================================
// Background slideshow
// =========================================================

let slides = [];             // [{name, image, caption, source, width, height}, ...]
let slideIndex = 0;
let slideTimer = null;
let visibleLayer = 0;        // which of the two layers is showing (0 or 1)
let slideshowId = 0;         // goes up each time a new slideshow starts
let kenBurnsTurn = 0;
let currentDestination;      // undefined = nothing loaded yet, null = default photos

// Download a photo first, so it never appears half-loaded
function loadImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = url;
  });
}

function renderDots(count) {
  captionDots.innerHTML = "";
  if (count < 2) return;
  for (let i = 0; i < count; i++) captionDots.appendChild(document.createElement("span"));
}

function updateDots(activeIndex) {
  captionDots.querySelectorAll("span").forEach((dot, i) => {
    dot.classList.remove("active");
    dot.classList.toggle("seen", i < activeIndex);
  });
  const active = captionDots.children[activeIndex];
  if (active) {
    void active.offsetWidth;  // restart the fill animation
    active.classList.add("active");
  }
}

async function showSlide(index, id) {
  const slide = slides[index];
  const ok = await loadImage(slide.image);
  if (id !== slideshowId) return;  // a newer slideshow started while we waited
  if (!ok) return;                 // broken photo: skip it, the next one comes in 5s

  // Put the new photo on the hidden layer, then fade it in over the visible one.
  // The photo is shown whole (never cut off); a blurred copy fills the rest of the frame.
  const next = layers[1 - visibleLayer];
  const current = layers[visibleLayer];
  const photo = next.querySelector(".gallery-photo");
  const fill = next.querySelector(".gallery-fill");

  photo.src = slide.image;
  fill.style.backgroundImage = `url("${slide.image.replace(/"/g, "%22")}")`;

  photo.classList.remove(...KEN_BURNS);
  void photo.offsetWidth;          // restart the movement animation
  photo.classList.add(KEN_BURNS[kenBurnsTurn++ % KEN_BURNS.length]);

  next.classList.add("visible");
  current.classList.remove("visible");
  visibleLayer = 1 - visibleLayer;
  updateDots(index);

  // Caption: textContent (not innerHTML) so names are shown as plain text
  caption.classList.add("hidden");
  setTimeout(() => {
    captionName.textContent = slide.name;
    captionText.textContent = slide.caption || "";
    if (slide.source) {
      captionCredit.href = slide.source;
      captionCredit.hidden = false;
    } else {
      captionCredit.hidden = true;
    }
    caption.classList.remove("hidden");
  }, 400);
}

function startSlideshow(attractions) {
  clearInterval(slideTimer);
  slideshowId += 1;
  const id = slideshowId;

  slides = attractions;
  slideIndex = 0;
  renderDots(slides.length);
  if (!slides.length) return;

  showSlide(0, id);
  if (slides.length > 1) {
    slideTimer = setInterval(() => {
      slideIndex = (slideIndex + 1) % slides.length;
      showSlide(slideIndex, id);
    }, SLIDE_EVERY_MS);
  }
}

// =========================================================
// Destination preview (calls GET /api/destination-preview)
// =========================================================

let typingTimer = null;
let lastPreviewText = null;
let previewRequestId = 0;

async function updatePreview(text) {
  text = text.trim();
  if (text === lastPreviewText) return;  // nothing changed
  lastPreviewText = text;

  const requestId = ++previewRequestId;
  let data;
  try {
    const response = await fetch(`/api/destination-preview?q=${encodeURIComponent(text)}`);
    data = await response.json();
  } catch (err) {
    return;  // keep whatever background we have
  }

  // The user kept typing and a newer request was sent: ignore this old answer
  if (requestId !== previewRequestId) return;

  // Same place as before (or default -> default): keep the slideshow running
  if (data.destination === currentDestination) return;
  currentDestination = data.destination;

  if (data.destination) {
    chip.textContent = `📍 ${data.destination}`;
    chip.hidden = false;
  } else {
    chip.hidden = true;
  }

  startSlideshow(data.attractions || []);
}

// Wait until the user pauses typing
input.addEventListener("input", () => {
  clearTimeout(typingTimer);
  typingTimer = setTimeout(() => updatePreview(input.value), TYPING_PAUSE_MS);
});

// =========================================================
// Progress steps
// =========================================================

const supervisorReason = document.getElementById("supervisor-reason");
const SUPERVISOR_HINT = supervisorReason.textContent;  // "Decides which agents to run"
const guardrailReason = document.getElementById("guardrail-reason");
const GUARDRAIL_HINT = guardrailReason.textContent;    // "Checks every request is safe"

const STATUS_TEXT = {
  waiting: "Waiting",
  active: "Working…",
  done: "Done",
  skipped: "Skipped",
  stopped: "Stopped",
  blocked: "Blocked",
};

// Shown under the guardrail box when it stops a request
const BLOCK_LABELS = {
  greeting: "Just a greeting, so no trip to plan yet",
  empty: "Empty request",
  too_long: "Request too long",
  off_topic: "Not a travel request",
  unsafe: "Unsafe request",
  injection: "Looks like a prompt-injection attempt",
};

// The agents the supervisor chose for this request, in order (always ending with final_agent)
let runOrder = [];

// Show one status on one agent's box
function setStatus(agent, status) {
  const li = progress.querySelector(`li[data-agent="${agent}"]`);
  if (!li) return;
  li.classList.remove("active", "done", "skipped", "blocked");
  if (["active", "done", "skipped", "blocked"].includes(status)) li.classList.add(status);
  li.querySelector(".agent-status").textContent = STATUS_TEXT[status];
}

// New request: everyone waits, the guardrail checks it first
function resetProgress() {
  runOrder = [];
  progress.querySelectorAll("li").forEach((li) => setStatus(li.dataset.agent, "waiting"));
  guardrailReason.textContent = GUARDRAIL_HINT;
  supervisorReason.textContent = SUPERVISOR_HINT;
  setStatus("guardrail_agent", "active");
}

// The guardrail allowed the request: now the supervisor reads it
function guardrailPassed() {
  setStatus("guardrail_agent", "done");
  guardrailReason.textContent = "Request looks safe";
  supervisorReason.textContent = "Reading your request…";
  setStatus("supervisor_agent", "active");
}

// The guardrail stopped the request: nobody else runs
function guardrailBlocked(category) {
  setStatus("guardrail_agent", category === "greeting" ? "done" : "blocked");
  guardrailReason.textContent = BLOCK_LABELS[category] || "Request stopped";
  progress.querySelectorAll("li").forEach((li) => {
    if (li.dataset.agent !== "guardrail_agent") setStatus(li.dataset.agent, "skipped");
  });
}

// The supervisor's plan arrived: show its reason, grey out skipped agents, start the first one
function applyPlan(agents, reason) {
  runOrder = [...agents, "final_agent"];
  setStatus("supervisor_agent", "done");
  supervisorReason.textContent = reason || SUPERVISOR_HINT;
  AGENTS.forEach((agent) => {
    if (!runOrder.includes(agent)) setStatus(agent, "skipped");
  });
  setStatus(runOrder[0], "active");
}

// An agent finished: mark it done and start the next one in the plan
function agentFinished(agent) {
  setStatus(agent, "done");
  const next = runOrder[runOrder.indexOf(agent) + 1];
  if (next) setStatus(next, "active");
}

// When something goes wrong, the agent that was working shows "Stopped"
function stopProgress() {
  const active = progress.querySelector("li.active");
  if (active) setStatus(active.dataset.agent, "stopped");
}

// =========================================================
// Showing the final plan
// =========================================================

let lastPlan = null;  // remembered for the PDF download

// Markdown -> safe HTML (DOMPurify removes anything unsafe, because the text comes from an AI)
function planToHtml(answer) {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(answer));
  return null;  // libraries didn't load (e.g. offline)
}

// A message from the guardrail instead of a plan (nothing to download)
function renderNotice(text, category) {
  lastPlan = null;
  const html = planToHtml(text);
  if (html !== null) resultBody.innerHTML = html;
  else resultBody.textContent = text;
  result.classList.add("notice");
  result.classList.toggle("notice-blocked", category !== "greeting");
  resultMeta.textContent = category === "greeting"
    ? ""
    : "Stopped by the guardrail before the supervisor or any tool ran.";
  downloadButton.hidden = true;
  result.hidden = false;
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderPlan(answer, llmCalls, message) {
  lastPlan = { answer, message, destination: currentDestination || null };
  result.classList.remove("notice", "notice-blocked");

  const html = planToHtml(answer);
  if (html !== null) {
    resultBody.innerHTML = html;
    resultBody.style.whiteSpace = "";
  } else {
    resultBody.textContent = answer;
    resultBody.style.whiteSpace = "pre-wrap";
  }
  const agentCount = runOrder.length || AGENTS.length;
  resultMeta.textContent =
    `Planned by the supervisor and ${agentCount} AI agent${agentCount === 1 ? "" : "s"} · ` +
    `${llmCalls} LLM calls · Prices are rough estimates, not live quotes.`;
  downloadButton.hidden = !window.html2pdf;  // hide the button if the PDF library didn't load
  result.hidden = false;
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(message) {
  stopProgress();
  errorBox.textContent = message;
  errorBox.hidden = false;
}

// =========================================================
// PDF download (made in the browser with html2pdf.js)
// =========================================================

// Builds a light, printer-friendly copy of the plan with a TripCrew AI header
function buildPdfDocument(plan) {
  const doc = document.createElement("div");
  doc.className = "pdf-doc";

  const today = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
  const header = document.createElement("div");
  header.className = "pdf-header";
  header.innerHTML = `
    <div>
      <div class="pdf-brand">TripCrew <span>AI</span></div>
      <div class="pdf-tagline">Your AI travel planner</div>
    </div>
    <div class="pdf-meta"><strong></strong><span></span></div>`;
  // textContent, so the destination name can't inject HTML
  header.querySelector(".pdf-meta strong").textContent = plan.destination ? `Trip to ${plan.destination}` : "Your travel plan";
  header.querySelector(".pdf-meta span").textContent = `Created ${today}`;
  doc.appendChild(header);

  const request = document.createElement("div");
  request.className = "pdf-request";
  request.textContent = `Your request: “${plan.message}”`;
  doc.appendChild(request);

  const body = document.createElement("div");
  const html = planToHtml(plan.answer);
  if (html !== null) {
    body.innerHTML = html;
  } else {
    body.textContent = plan.answer;
    body.style.whiteSpace = "pre-wrap";
  }
  doc.appendChild(body);

  const disclaimer = document.createElement("p");
  disclaimer.className = "pdf-disclaimer";
  disclaimer.textContent =
    "Generated by TripCrew AI. Prices are rough estimates, not live quotes. " +
    "Flight data shows live status, not fares. Please confirm details before booking.";
  doc.appendChild(disclaimer);

  return doc;
}

async function downloadPdf() {
  if (!lastPlan || !window.html2pdf) return;

  downloadButton.disabled = true;
  downloadButton.textContent = "Preparing PDF…";

  const name = (lastPlan.destination || "trip").replace(/[^a-z0-9]+/gi, "-");
  const options = {
    margin: [14, 14, 18, 14],                        // mm: top, left, bottom, right
    filename: `TripCrew-AI-${name}-plan.pdf`,
    image: { type: "jpeg", quality: 0.95 },
    html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
    jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
    // Don't cut these in half at the bottom of a page
    pagebreak: { mode: ["css", "legacy"], avoid: ["tr", "li", "h2", "h3", "blockquote", ".pdf-header"] },
  };

  try {
    await html2pdf()
      .set(options)
      .from(buildPdfDocument(lastPlan))
      .toPdf()
      .get("pdf")
      .then((pdf) => {
        // "TripCrew AI" footer and page number on every page
        const pages = pdf.internal.getNumberOfPages();
        const width = pdf.internal.pageSize.getWidth();
        const height = pdf.internal.pageSize.getHeight();
        for (let i = 1; i <= pages; i++) {
          pdf.setPage(i);
          pdf.setFontSize(9);
          pdf.setTextColor(120);
          pdf.text("TripCrew AI · Your AI travel planner", 14, height - 8);
          pdf.text(`Page ${i} of ${pages}`, width - 14, height - 8, { align: "right" });
        }
      })
      .save();
  } catch (err) {
    showError("Sorry, the PDF could not be created. Please try again.");
  } finally {
    downloadButton.disabled = false;
    downloadButton.textContent = "⬇ Download PDF";
  }
}

downloadButton.addEventListener("click", downloadPdf);

// =========================================================
// Planning the trip (calls POST /api/travel-stream)
// =========================================================

async function planTrip(message) {
  button.disabled = true;
  button.textContent = "Planning…";
  errorBox.hidden = true;
  result.hidden = true;
  resetProgress();

  try {
    const response = await fetch("/api/travel-stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!response.ok || !response.body) throw new Error(`Server error ${response.status}`);

    // The server sends one line of JSON each time an agent finishes
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;

    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();  // last piece may be an incomplete line

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);

        if (event.type === "step" && event.agent === "guardrail_agent") {
          guardrailPassed();
        } else if (event.type === "blocked") {
          guardrailBlocked(event.category);
          renderNotice(event.message, event.category);
          finished = true;
        } else if (event.type === "plan") {
          // The supervisor decided which agents to run
          applyPlan(event.agents, event.reason);
        } else if (event.type === "step") {
          // This agent finished, so the next one in the plan is now working
          agentFinished(event.agent);
        } else if (event.type === "done") {
          agentFinished("final_agent");
          renderPlan(event.answer, event.llm_calls, message);
          finished = true;
        } else if (event.type === "error") {
          showError(event.message);
          finished = true;
        }
      }
    }

    if (!finished) showError("The connection closed before the plan was ready. Please try again.");
  } catch (err) {
    showError("Could not reach the TripCrew server. Is it still running?");
  } finally {
    button.disabled = false;
    button.textContent = "Plan my trip";
  }
}

// In the text box: Enter = plan the trip, Shift+Enter = new line
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

// Enter key or button click
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  clearTimeout(typingTimer);
  updatePreview(message);  // switch background right away
  planTrip(message);
});

// =========================================================
// Start: show the default travel photos
// =========================================================

updatePreview("");
