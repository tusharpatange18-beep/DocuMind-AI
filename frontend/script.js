const FALLBACK_API_BASE = "http://127.0.0.1:8000";
let API_BASE = FALLBACK_API_BASE;

function normalizeBase(u) {
  return (u || "").trim().replace(/\/+$/, "");
}

function getConfiguredApiBase() {
  const qs = new URLSearchParams(window.location.search);
  const fromQuery = qs.get("api");
  const fromMeta = document.querySelector('meta[name="api-base"]')?.getAttribute("content");
  return normalizeBase(window.API_BASE || fromQuery || fromMeta || "");
}

async function initApiBase() {
  const configured = getConfiguredApiBase();
  if (configured) {
    API_BASE = configured;
    return;
  }

  // If the frontend is served by the backend (same origin), use relative calls.
  // Use a short timeout so misconfiguration doesn't feel like "slow upload".
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 800);
    const res = await fetch("/", { method: "GET", signal: controller.signal });
    clearTimeout(timer);
    const data = await res.json().catch(() => null);
    if (res.ok && data && typeof data.status === "string" && data.status.includes("DocuMind")) {
      API_BASE = "";
      return;
    }
  } catch (_) {
    // ignore
  }

  API_BASE = FALLBACK_API_BASE;
}

const el = (id) => document.getElementById(id);

const sidebar = el("sidebar");
const overlay = el("overlay");
const menuBtn = el("menuBtn");

const greeting = el("greeting");
const chatArea = el("chatArea");

const pdfInput = el("pdfInput");
const uploadBtn = el("uploadBtn");
const uploadStatus = el("uploadStatus");

const rebuildBtn = el("rebuildBtn");
const rebuildStatus = el("rebuildStatus");

const clearHistoryBtn = el("clearHistoryBtn");
const clearStatus = el("clearStatus");

const promptUpload = el("promptUpload");
const promptExample = el("promptExample");

const questionInput = el("questionInput");
const sendBtn = el("sendBtn");

function isMobile() {
  return window.matchMedia("(max-width: 900px)").matches;
}

function openSidebar() {
  if (!isMobile()) return;
  sidebar.classList.add("open");
  overlay.classList.add("show");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  overlay.classList.remove("show");
}

menuBtn.addEventListener("click", () => {
  if (sidebar.classList.contains("open")) closeSidebar();
  else openSidebar();
});

overlay.addEventListener("click", closeSidebar);

promptUpload.addEventListener("click", () => {
  openSidebar();
  setTimeout(() => pdfInput.click(), 50);
});

promptExample.addEventListener("click", () => {
  questionInput.value = "Summarize the key points in the document.";
  questionInput.focus();
});

function escapeHtml(s) {
  return s.replace(/[&<>'\"]/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    "\"": "&quot;",
  }[c]));
}

function renderEmptyHint() {
  chatArea.innerHTML = "<div class=\"empty-hint\">📄 Upload a document in the sidebar and start asking questions!</div>";
}

function renderMessages(messages) {
  chatArea.innerHTML = "";

  if (!messages || messages.length === 0) {
    renderEmptyHint();
    return;
  }

  for (const item of messages) {
    addUserMessage(item.question);
    addAiMessage(item.answer, item.sources);
  }

  scrollToBottom();
}

function addUserMessage(text) {
  const html = `
    <div class="message-group user">
      <div class="message-content">
        <div class="message-bubble user">${escapeHtml(text)}</div>
      </div>
    </div>
  `;
  chatArea.insertAdjacentHTML("beforeend", html);
}

function sourcesText(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return "";
  const top = sources.slice(0, 2);
  const labels = top.map((s) => `${s.source || "Document"} (page ${s.page ?? "N/A"})`);
  return `📚 Sources: ${labels.join(", ")}`;
}

function addAiMessage(text, sources) {
  const src = sourcesText(sources);
  const html = `
    <div class="message-group">
      <div class="message-avatar ai-avatar-small"></div>
      <div class="message-content">
        <div class="message-bubble ai">${escapeHtml(text)}</div>
        ${src ? `<div class="sources-info">${escapeHtml(src)}</div>` : ""}
      </div>
    </div>
  `;
  chatArea.insertAdjacentHTML("beforeend", html);
}

function addThinkingMessage() {
  const html = `
    <div class="message-group" id="thinkingRow">
      <div class="message-avatar ai-avatar-small"></div>
      <div class="message-content">
        <div class="message-bubble ai">
          <span class="thinking"><span class="spinner"></span>Thinking…</span>
        </div>
      </div>
    </div>
  `;
  chatArea.insertAdjacentHTML("beforeend", html);
  scrollToBottom();
}

function replaceThinking(answer, sources) {
  const row = el("thinkingRow");
  if (row) row.remove();
  addAiMessage(answer, sources);
  scrollToBottom();
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function apiJson(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.detail || `Request failed: ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

async function loadHistory() {
  const history = await apiJson("/history", { method: "GET" });
  greeting.style.display = (history && history.length) ? "none" : "block";
  renderMessages(history);
}

uploadBtn.addEventListener("click", async () => {
  uploadStatus.textContent = "";

  const file = pdfInput.files && pdfInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Please choose a PDF file.";
    return;
  }

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    uploadStatus.textContent = "Only PDF files are supported.";
    return;
  }

  uploadStatus.textContent = "Uploading… 0%";

  const form = new FormData();
  form.append("file", file);

  try {
    const data = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/upload-pdf`, true);

      xhr.upload.onprogress = (evt) => {
        if (!evt.lengthComputable) {
          uploadStatus.textContent = "Uploading…";
          return;
        }
        const pct = Math.floor((evt.loaded / evt.total) * 100);
        const mb = (evt.loaded / (1024 * 1024)).toFixed(1);
        const totalMb = (evt.total / (1024 * 1024)).toFixed(1);
        uploadStatus.textContent = `Uploading… ${pct}% (${mb}/${totalMb} MB)`;
      };

      xhr.onerror = () => reject(new Error("Network error during upload"));

      xhr.onload = () => {
        let parsed = {};
        try {
          parsed = JSON.parse(xhr.responseText || "{}");
        } catch (_) {
          parsed = {};
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(parsed);
        else reject(new Error(parsed.detail || `Upload failed: ${xhr.status}`));
      };

      xhr.send(form);
    });

    uploadStatus.textContent = `✅ Uploaded: ${data.filename || file.name}`;
    if (isMobile()) closeSidebar();
  } catch (e) {
    uploadStatus.textContent = `❌ ${e.message}`;
  }
});

rebuildBtn.addEventListener("click", async () => {
  rebuildStatus.textContent = "Processing documents…";
  try {
    await apiJson("/rebuild-knowledge-base", { method: "POST" });
    rebuildStatus.textContent = "✅ Knowledge base rebuilt";
    if (isMobile()) closeSidebar();
  } catch (e) {
    rebuildStatus.textContent = `❌ ${e.message}`;
  }
});

clearHistoryBtn.addEventListener("click", async () => {
  clearStatus.textContent = "Clearing…";
  try {
    await apiJson("/history", { method: "DELETE" });
    clearStatus.textContent = "✅ Cleared";
    greeting.style.display = "block";
    renderMessages([]);
    if (isMobile()) closeSidebar();
  } catch (e) {
    clearStatus.textContent = `❌ ${e.message}`;
  }
});

async function askQuestion() {
  const q = (questionInput.value || "").trim();
  if (!q) return;

  greeting.style.display = "none";

  addUserMessage(q);
  questionInput.value = "";
  addThinkingMessage();

  try {
    const data = await apiJson("/ask", {
      method: "POST",
      body: JSON.stringify({ question: q }),
    });

    replaceThinking(data.answer || "", data.sources || []);
  } catch (e) {
    replaceThinking(`❌ ${e.message}`, []);
  }
}

sendBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    askQuestion();
  }
});

window.addEventListener("load", async () => {
  try {
    await initApiBase();

    await loadHistory();
  } catch (e) {
    greeting.style.display = "block";
    renderEmptyHint();
  }
});
