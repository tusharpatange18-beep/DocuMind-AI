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
  } catch (_) {}
  API_BASE = FALLBACK_API_BASE;
}

const el = (id) => document.getElementById(id);

// Elements
const sidebar = el("sidebar");
const toggleSidebarBtn = el("toggleSidebarBtn");
const closeSidebarMobile = el("closeSidebarMobile");
const sidebarOverlay = el("sidebarOverlay");
const newChatBtn = el("newChatBtn");

const headerUploadBtn = el("headerUploadBtn");
const headerRebuildBtn = el("headerRebuildBtn");
const pdfInput = el("pdfInput");
const topStatusBar = el("topStatusBar");

const welcomeState = el("welcomeState");
const chatArea = el("chatArea");

const questionInput = el("questionInput");
const sendBtn = el("sendBtn");
const inputContext = el("inputContext");
const inputContextText = el("inputContextText");

const settingsMenuBtn = el("settingsMenuBtn");
const settingsModal = el("settingsModal");
const closeSettingsBtn = el("closeSettingsBtn");
const clearHistoryBtn = el("clearHistoryBtn");
const sidebarSearch = el("sidebarSearch");
const sidebarHistory = el("sidebarHistory");

// Conversation Manager
const StorageKey = "documind_sessions";
let globalHistory = [];

class ConversationManager {
  constructor() {
    this.sessions = JSON.parse(localStorage.getItem(StorageKey) || "[]");
    this.activeSessionId = null;
  }
  
  save() {
    localStorage.setItem(StorageKey, JSON.stringify(this.sessions));
  }
  
  syncWithGlobal(globalArray) {
    globalHistory = globalArray || [];
    
    // If backend was cleared, clear our sessions
    if (globalHistory.length === 0) {
      this.sessions = [];
      this.activeSessionId = null;
      this.save();
      return;
    }
    
    // If backend has messages but we have no sessions, create one to adopt them all
    if (this.sessions.length === 0 && globalHistory.length > 0) {
      this.sessions.push({
        id: Date.now().toString(),
        title: globalHistory[0].question || "New Chat",
        startIndex: 0,
        count: globalHistory.length,
        timestamp: Date.now()
      });
      this.activeSessionId = this.sessions[0].id;
      this.save();
    }
    
    // Validate bounds: The active session should adopt any new messages that arrived
    if (this.activeSessionId) {
      const active = this.sessions.find(s => s.id === this.activeSessionId);
      if (active) {
        let maxEnd = 0;
        this.sessions.forEach(s => {
           maxEnd = Math.max(maxEnd, s.startIndex + s.count);
        });
        if (globalHistory.length > maxEnd) {
           active.count += (globalHistory.length - maxEnd);
           this.save();
        }
      }
    }
  }

  getActiveMessages() {
    if (!this.activeSessionId) return [];
    const active = this.sessions.find(s => s.id === this.activeSessionId);
    if (!active) return [];
    return globalHistory.slice(active.startIndex, active.startIndex + active.count);
  }
  
  createNewSession() {
    const startIndex = globalHistory.length;
    const newSession = {
      id: Date.now().toString(),
      title: "New Chat",
      startIndex: startIndex,
      count: 0,
      timestamp: Date.now()
    };
    this.sessions.unshift(newSession);
    this.activeSessionId = newSession.id;
    this.save();
  }
  
  switchSession(id) {
    const session = this.sessions.find(s => s.id === id);
    if (session) {
      this.activeSessionId = id;
      return true;
    }
    return false;
  }
  
  renameSession(id, newTitle) {
    const session = this.sessions.find(s => s.id === id);
    if (session) {
      session.title = newTitle;
      this.save();
    }
  }
  
  deleteSession(id) {
    this.sessions = this.sessions.filter(s => s.id !== id);
    if (this.activeSessionId === id) {
      this.activeSessionId = this.sessions.length > 0 ? this.sessions[0].id : null;
    }
    this.save();
  }
  
  updateTitleIfNew(question) {
    const active = this.sessions.find(s => s.id === this.activeSessionId);
    if (active && active.count === 0 && active.title === "New Chat") {
       active.title = question.length > 30 ? question.substring(0, 30) + "..." : question;
       this.save();
    }
  }
}

const chatManager = new ConversationManager();

// Sidebar Toggle Logic
function isMobile() {
  return window.innerWidth <= 768;
}

function toggleSidebar() {
  if (isMobile()) {
    sidebar.classList.toggle("open");
    sidebarOverlay.classList.toggle("show");
  } else {
    sidebar.classList.toggle("closed");
  }
}

toggleSidebarBtn.addEventListener("click", toggleSidebar);
closeSidebarMobile.addEventListener("click", toggleSidebar);
sidebarOverlay.addEventListener("click", toggleSidebar);

// Settings Modal
settingsMenuBtn.addEventListener("click", () => {
  settingsModal.style.display = "flex";
});
closeSettingsBtn.addEventListener("click", () => {
  settingsModal.style.display = "none";
});

function setStatus(msg, isError = false) {
  if (!msg) {
    topStatusBar.style.display = "none";
    return;
  }
  topStatusBar.textContent = msg;
  topStatusBar.style.color = isError ? "var(--error-color)" : "var(--text-secondary)";
  topStatusBar.style.display = "flex";
}

// API Wrapper
async function apiJson(path, options = {}) {
  const timeoutMs = (path === "/ask") ? 125000 : 30000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    signal: controller.signal,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  }).finally(() => clearTimeout(timer));
  
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed: ${res.status}`);
  }
  return data;
}

// UI Rendering
function escapeHtml(s) {
  if (!s) return "";
  return s.toString().replace(/[&<>'\"]/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;"
  }[c]));
}

function scrollToBottom() {
  const viewport = document.querySelector('.chat-viewport');
  if (viewport) {
    viewport.scrollTop = viewport.scrollHeight;
  }
}

function renderMessages(messages) {
  chatArea.innerHTML = "";
  if (!messages || messages.length === 0) {
    welcomeState.style.display = "flex";
    return;
  }
  welcomeState.style.display = "none";
  for (const item of messages) {
    addUserMessage(item.question);
    addAiMessage(item.answer, item.sources);
  }
  scrollToBottom();
}

// Markdown Configuration
if (window.marked) {
  marked.setOptions({
    highlight: function(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(code, { language: lang }).value;
        } catch (__) {}
      }
      return code; // use external default escaping
    }
  });

  const renderer = new marked.Renderer();
  renderer.code = function(code, language) {
    const validLang = !!(language && hljs.getLanguage(language));
    const langLabel = validLang ? language : "code";
    const highlighted = validLang ? hljs.highlight(code, { language }).value : escapeHtml(code);
    
    // We escape the raw code for the copy button data attribute
    const escapedRaw = escapeHtml(code);

    return `
      <div class="code-block-wrapper">
        <div class="code-block-header">
          <span>${langLabel}</span>
          <button class="code-copy-btn" onclick="copyTextToClipboard(this, decodeURIComponent('${encodeURIComponent(code)}'))">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>Copy</span>
          </button>
        </div>
        <pre><code class="hljs ${language || ''}">${highlighted}</code></pre>
      </div>
    `;
  };
  marked.use({ renderer });
}

async function copyTextToClipboard(btnElement, text) {
  try {
    await navigator.clipboard.writeText(text);
    const span = btnElement.querySelector('span');
    const originalText = span.textContent;
    span.textContent = 'Copied!';
    setTimeout(() => { span.textContent = originalText; }, 2000);
  } catch (err) {
    console.error('Failed to copy', err);
  }
}

function addUserMessage(text) {
  welcomeState.style.display = "none";
  const html = `
    <div class="msg-row user">
      <div class="msg-content msg-bubble-user">
        ${escapeHtml(text)}
      </div>
    </div>
  `;
  chatArea.insertAdjacentHTML("beforeend", html);
}

function renderSourcesHtml(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return "";
  const top = sources.slice(0, 3);
  const pills = top.map(s => {
    const label = `${s.source || "doc"} · Page ${s.page ?? "?"}`;
    return `<div class="source-ref"><span class="source-icon">📄</span> <span>${escapeHtml(label)}</span></div>`;
  }).join("");
  return `
    <div class="msg-sources-section">
      <div class="sources-header">Sources</div>
      <div class="msg-sources">${pills}</div>
    </div>
  `;
}

function addAiMessage(text, sources, isError = false) {
  let contentHtml = "";
  
  if (isError) {
    contentHtml = `<div class="msg-error-box">${escapeHtml(text)}</div>`;
  } else {
    // Parse markdown and sanitize
    const rawHtml = window.marked ? marked.parse(text) : escapeHtml(text);
    const cleanHtml = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;
    const sourcesHtml = renderSourcesHtml(sources);
    contentHtml = `<div class="markdown-body">${cleanHtml}</div>${sourcesHtml}`;
  }

  const encodedRawText = encodeURIComponent(text);

  const html = `
    <div class="msg-row ai">
      <div class="msg-content msg-box-ai">
        <div class="ai-avatar">🤖</div>
        <div class="ai-text">
          ${contentHtml}
          <div class="msg-actions">
            ${isError 
              ? `<button class="action-btn" onclick="retryLastMessage()">
                   <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                   Retry
                 </button>`
              : `<button class="action-btn" onclick="copyTextToClipboard(this, decodeURIComponent('${encodedRawText}'))">
                   <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                   Copy
                 </button>`
            }
          </div>
        </div>
      </div>
    </div>
  `;
  chatArea.insertAdjacentHTML("beforeend", html);
}

function addThinkingMessage() {
  const html = `
    <div class="msg-row ai" id="thinkingRow">
      <div class="msg-content msg-box-ai">
        <div class="ai-avatar">🤖</div>
        <div class="ai-text">
          <span class="thinking-text"><div class="spinner-mini"></div> Thinking...</span>
        </div>
      </div>
    </div>
  `;
  chatArea.insertAdjacentHTML("beforeend", html);
  scrollToBottom();
}

function replaceThinking(answer, sources, isError = false) {
  const row = el("thinkingRow");
  if (row) row.remove();
  addAiMessage(answer, sources, isError);
  scrollToBottom();
}

// Data Actions
function renderSidebar() {
  sidebarHistory.innerHTML = "";
  
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterday = today - 86400000;
  const prev7 = today - (7 * 86400000);
  
  const groups = {
    "Today": [],
    "Yesterday": [],
    "Previous 7 Days": [],
    "Older": []
  };
  
  const searchQuery = sidebarSearch.value.toLowerCase();
  
  chatManager.sessions.forEach(s => {
    if (searchQuery && !s.title.toLowerCase().includes(searchQuery)) return;
    if (s.timestamp >= today) groups["Today"].push(s);
    else if (s.timestamp >= yesterday) groups["Yesterday"].push(s);
    else if (s.timestamp >= prev7) groups["Previous 7 Days"].push(s);
    else groups["Older"].push(s);
  });
  
  for (const [header, sessions] of Object.entries(groups)) {
    if (sessions.length === 0) continue;
    
    const groupEl = document.createElement("div");
    groupEl.className = "history-group";
    groupEl.innerHTML = `<div class="history-header">${header}</div>`;
    
    sessions.forEach(s => {
      const itemEl = document.createElement("div");
      itemEl.className = `history-item ${s.id === chatManager.activeSessionId ? "active" : ""}`;
      itemEl.onclick = (e) => {
        if (e.target.closest('.history-actions') || e.target.closest('input')) return;
        chatManager.switchSession(s.id);
        renderMessages(chatManager.getActiveMessages());
        renderSidebar();
        if (isMobile()) toggleSidebar();
      };
      
      const titleSpan = document.createElement("span");
      titleSpan.className = "history-title";
      titleSpan.textContent = s.title;
      
      const actionsEl = document.createElement("div");
      actionsEl.className = "history-actions";
      
      const renameBtn = document.createElement("button");
      renameBtn.className = "history-action-btn";
      renameBtn.innerHTML = "✎";
      renameBtn.title = "Rename";
      renameBtn.onclick = (e) => {
        e.stopPropagation();
        itemEl.innerHTML = "";
        const input = document.createElement("input");
        input.className = "history-title-input";
        input.value = s.title;
        itemEl.appendChild(input);
        input.focus();
        input.onblur = input.onkeydown = (ev) => {
          if (ev.type === 'keydown' && ev.key !== 'Enter') return;
          if (input.value.trim()) {
            chatManager.renameSession(s.id, input.value.trim());
          }
          renderSidebar();
        };
      };
      
      const deleteBtn = document.createElement("button");
      deleteBtn.className = "history-action-btn";
      deleteBtn.innerHTML = "✕";
      deleteBtn.title = "Delete";
      deleteBtn.onclick = (e) => {
        e.stopPropagation();
        chatManager.deleteSession(s.id);
        renderSidebar();
        renderMessages(chatManager.getActiveMessages());
      };
      
      actionsEl.appendChild(renameBtn);
      actionsEl.appendChild(deleteBtn);
      
      itemEl.appendChild(titleSpan);
      itemEl.appendChild(actionsEl);
      groupEl.appendChild(itemEl);
    });
    
    sidebarHistory.appendChild(groupEl);
  }
}

sidebarSearch.addEventListener("input", renderSidebar);

async function loadHistory() {
  try {
    const history = await apiJson("/history", { method: "GET" });
    chatManager.syncWithGlobal(history);
    
    if (!chatManager.activeSessionId) {
      chatManager.createNewSession();
    }
    
    renderSidebar();
    renderMessages(chatManager.getActiveMessages());
  } catch (e) {
    console.error("Failed to load history", e);
  }
}

headerUploadBtn.addEventListener("click", () => {
  pdfInput.click();
});

pdfInput.addEventListener("change", async () => {
  const file = pdfInput.files && pdfInput.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setStatus("Only PDF files are supported.", true);
    return;
  }

  setStatus(`Uploading ${file.name}... 0%`);
  headerUploadBtn.disabled = true;
  
  const form = new FormData();
  form.append("file", file);

  try {
    const data = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/upload-pdf`, true);

      xhr.upload.onprogress = (evt) => {
        if (evt.lengthComputable) {
          const pct = Math.floor((evt.loaded / evt.total) * 100);
          setStatus(`Uploading ${file.name}... ${pct}%`);
        }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.onload = () => {
        let parsed = {};
        try { parsed = JSON.parse(xhr.responseText || "{}"); } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300) resolve(parsed);
        else reject(new Error(parsed.detail || `Upload failed: ${xhr.status}`));
      };
      xhr.send(form);
    });

    setStatus(`Uploaded: ${data.filename || file.name}`);
    
    // Show document context indicator
    inputContext.style.display = "flex";
    inputContextText.textContent = data.filename || file.name;
    
    setTimeout(() => setStatus(""), 3000);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    headerUploadBtn.disabled = false;
    pdfInput.value = "";
  }
});

headerRebuildBtn.addEventListener("click", async () => {
  setStatus("Rebuilding knowledge base...");
  headerRebuildBtn.disabled = true;
  try {
    await apiJson("/rebuild-knowledge-base", { method: "POST" });
    setStatus("Knowledge base rebuilt successfully.");
    setTimeout(() => setStatus(""), 3000);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    headerRebuildBtn.disabled = false;
  }
});

clearHistoryBtn.addEventListener("click", async () => {
  clearHistoryBtn.disabled = true;
  try {
    await apiJson("/history", { method: "DELETE" });
    chatManager.syncWithGlobal([]);
    chatManager.createNewSession();
    renderSidebar();
    renderMessages([]);
    settingsModal.style.display = "none";
  } catch (e) {
    alert("Failed to clear: " + e.message);
  } finally {
    clearHistoryBtn.disabled = false;
  }
});

newChatBtn.addEventListener("click", () => {
  chatManager.createNewSession();
  renderSidebar();
  renderMessages([]);
  inputContext.style.display = "none";
  if (isMobile()) toggleSidebar();
});

// Chat Input
function adjustTextareaHeight() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 200) + "px";
}

function updateSendBtnState() {
  sendBtn.disabled = questionInput.value.trim().length === 0;
}

questionInput.addEventListener("input", () => {
  adjustTextareaHeight();
  updateSendBtnState();
});

let lastQuestion = "";

async function handleSend(retryQuestion = null) {
  const q = retryQuestion || questionInput.value.trim();
  if (!q) return;

  if (!retryQuestion) {
    chatManager.updateTitleIfNew(q);
    renderSidebar();
    addUserMessage(q);
    questionInput.value = "";
    adjustTextareaHeight();
    updateSendBtnState();
  }
  
  lastQuestion = q;
  addThinkingMessage();
  questionInput.disabled = true;
  sendBtn.disabled = true;

  try {
    const data = await apiJson("/ask", {
      method: "POST",
      body: JSON.stringify({ question: q }),
    });
    
    // Add to global array and sync count
    globalHistory.push({ question: q, answer: data.answer || "", sources: data.sources || [] });
    const active = chatManager.sessions.find(s => s.id === chatManager.activeSessionId);
    if (active) {
      active.count += 1;
      chatManager.save();
    }
    
    replaceThinking(data.answer || "", data.sources || []);
  } catch (e) {
    replaceThinking(`Error: ${e.message}`, [], true);
  } finally {
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.focus();
    updateSendBtnState();
  }
}

function retryLastMessage() {
  if (lastQuestion) {
    // Remove the error message visually
    const rows = chatArea.querySelectorAll('.msg-row.ai');
    if (rows.length > 0) {
      rows[rows.length - 1].remove();
    }
    handleSend(lastQuestion);
  }
}

sendBtn.addEventListener("click", handleSend);

questionInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    if (!sendBtn.disabled) handleSend();
  }
});

window.addEventListener("load", async () => {
  try {
    await initApiBase();
    await loadHistory();
  } catch (e) {
    console.error("Init failed", e);
  }
});
