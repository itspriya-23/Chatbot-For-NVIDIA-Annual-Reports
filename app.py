"""
NVIDIA Financial RAG Chatbot - Web Application
Zero-dependency, production-grade web application powered by Python http.server.
Features a modern, responsive chat interface with verifiable page citations,
expandable source evidence drawers, and pre-configured financial queries.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.rag_engine import RAGEngine

# Initialize RAG Engine globally
engine = RAGEngine()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NVIDIA Annual Reports (1998-2026) AI Chatbot</title>
  <style>
    :root {
      --bg-dark: #0f1117;
      --bg-card: #181b24;
      --bg-chat-user: #232838;
      --bg-chat-bot: #1c2130;
      --nv-green: #76b900;
      --nv-green-hover: #8ae000;
      --border: #2a3142;
      --text-main: #f0f3f8;
      --text-muted: #9aa5b8;
      --badge-bg: rgba(118, 185, 0, 0.15);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background: var(--bg-card);
      border-bottom: 1px solid var(--border);
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .logo-badge {
      background: var(--nv-green);
      color: #000;
      font-weight: 900;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 14px;
      letter-spacing: 0.5px;
    }
    .brand h1 { font-size: 18px; font-weight: 700; }
    .brand span { font-size: 13px; color: var(--text-muted); }
    .meta-tags {
      display: flex;
      gap: 10px;
    }
    .tag {
      background: var(--badge-bg);
      color: var(--nv-green);
      border: 1px solid rgba(118,185,0,0.3);
      padding: 4px 10px;
      border-radius: 14px;
      font-size: 12px;
      font-weight: 600;
    }
    .main-container {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    .sidebar {
      width: 320px;
      background: var(--bg-card);
      border-right: 1px solid var(--border);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      overflow-y: auto;
    }
    .sidebar h3 {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .example-btn {
      background: var(--bg-chat-user);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 13px;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s;
      line-height: 1.4;
      margin-bottom: 8px;
      width: 100%;
    }
    .example-btn:hover {
      border-color: var(--nv-green);
      background: #2b3245;
    }
    .chat-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      position: relative;
    }
    .messages-box {
      flex: 1;
      overflow-y: auto;
      padding: 24px 32px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .msg-row {
      display: flex;
      flex-direction: column;
      max-width: 85%;
    }
    .msg-user {
      align-self: flex-end;
    }
    .msg-bot {
      align-self: flex-start;
      max-width: 90%;
    }
    .bubble {
      padding: 14px 18px;
      border-radius: 10px;
      line-height: 1.6;
      font-size: 14.5px;
    }
    .bubble-user {
      background: var(--bg-chat-user);
      border: 1px solid #3b4259;
      color: #fff;
    }
    .bubble-bot {
      background: var(--bg-chat-bot);
      border: 1px solid var(--border);
      color: var(--text-main);
    }
    .bubble-bot p { margin-bottom: 10px; }
    .bubble-bot ul, .bubble-bot ol { margin-left: 20px; margin-bottom: 10px; }
    .sources-container {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px dashed var(--border);
    }
    .sources-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--nv-green);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .source-item {
      background: #141822;
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-bottom: 6px;
      overflow: hidden;
      font-size: 12.5px;
    }
    .source-header {
      padding: 8px 12px;
      display: flex;
      justify-content: space-between;
      cursor: pointer;
      background: rgba(255,255,255,0.02);
    }
    .source-header:hover { background: rgba(255,255,255,0.05); }
    .source-score {
      color: var(--nv-green);
      font-weight: 600;
    }
    .source-body {
      padding: 10px 12px;
      display: none;
      color: var(--text-muted);
      border-top: 1px solid var(--border);
      white-space: pre-wrap;
      font-family: monospace;
      font-size: 11.5px;
      background: #0d1017;
    }
    .input-bar {
      padding: 16px 32px;
      background: var(--bg-card);
      border-top: 1px solid var(--border);
      display: flex;
      gap: 12px;
    }
    .input-bar input {
      flex: 1;
      background: var(--bg-dark);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      color: var(--text-main);
      font-size: 15px;
      outline: none;
    }
    .input-bar input:focus { border-color: var(--nv-green); }
    .send-btn {
      background: var(--nv-green);
      color: #000;
      border: none;
      font-weight: 700;
      padding: 0 24px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      transition: background 0.2s;
    }
    .send-btn:hover { background: var(--nv-green-hover); }
    .send-btn:disabled { background: #445; cursor: not-allowed; }
    .abstention-banner {
      background: rgba(255, 170, 0, 0.1);
      border-left: 3px solid #ffaa00;
      padding: 8px 12px;
      border-radius: 4px;
      color: #ffbb33;
      font-size: 13px;
      margin-bottom: 8px;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo-badge">NVIDIA</div>
      <div>
        <h1>Annual Reports & Proxy Statement RAG Chatbot</h1>
        <span>Official SEC 10-K & DEF 14A Filing Intelligence (1998 – 2026)</span>
      </div>
    </div>
    <div class="meta-tags">
      <div class="tag">11 Filings (1998-2026)</div>
      <div class="tag">3072-dim Vectors</div>
      <div class="tag">Gemini 3.5 Flash</div>
    </div>
  </header>

  <div class="main-container">
    <div class="sidebar">
      <div>
        <h3>Benchmark Financial Queries</h3>
        <button class="example-btn" onclick="askQuestion('What was NVIDIA\\'s total revenue in fiscal year 2024?')">
          💰 What was NVIDIA's total revenue in FY2024?
        </button>
        <button class="example-btn" onclick="askQuestion('What was NVIDIA\\'s Data Center segment revenue for fiscal year 2024?')">
          🏢 What was Data Center revenue in FY2024?
        </button>
        <button class="example-btn" onclick="askQuestion('What was NVIDIA\\'s GAAP gross margin in FY2024 compared to FY2023?')">
          📈 How did GAAP gross margin change from FY23 to FY24?
        </button>
        <button class="example-btn" onclick="askQuestion('Who was President and CEO according to NVIDIA\\'s 1999 proxy statement?')">
          📜 Who was CEO in NVIDIA's 1999 proxy statement?
        </button>
        <button class="example-btn" onclick="askQuestion('What major acquisition was discussed in the fiscal year 2020 annual report?')">
          🤝 What acquisition was discussed in FY2020?
        </button>
        <button class="example-btn" onclick="askQuestion('What was NVIDIA\\'s total net income in fiscal year 1982?')">
          🚫 What was net income in 1982? (Tests Abstention)
        </button>
      </div>

      <div>
        <h3>Ingested Documents</h3>
        <ul style="font-size: 12px; color: var(--text-muted); list-style: disc; margin-left: 16px; line-height: 1.8;">
          <li>NVIDIA 2026 Annual Report (10-K)</li>
          <li>NVIDIA 2025 Annual Report (10-K)</li>
          <li>NVIDIA 2024 Annual Report (10-K)</li>
          <li>NVIDIA 2023 Annual Report (10-K)</li>
          <li>NVIDIA 2020 Annual Report (10-K)</li>
          <li>NVIDIA 2018 Annual Report (10-K)</li>
          <li>NVIDIA 2015 Annual Report (10-K)</li>
          <li>NVIDIA 2008 Proxy (DEF 14A)</li>
          <li>NVIDIA 2004 Proxy (DEF 14A)</li>
          <li>NVIDIA 1999 Proxy (DEF 14A)</li>
          <li>NVIDIA 1998 Proxy (DEF 14A)</li>
        </ul>
      </div>
    </div>

    <div class="chat-area">
      <div class="messages-box" id="messagesBox">
        <div class="msg-row msg-bot">
          <div class="bubble bubble-bot">
            <p><strong>Welcome to the NVIDIA Annual Reports & Proxies RAG Assistant!</strong></p>
            <p>I can answer detailed financial, structural, executive, and historical questions grounded in NVIDIA's official Form 10-K Annual Reports and Proxy Statements covering <strong>1998 through 2026</strong>.</p>
            <p>Every answer includes <strong>verified source references with exact page numbers</strong>, and I will explicitly abstain if evidence is missing.</p>
          </div>
        </div>
      </div>

      <form class="input-bar" id="chatForm" onsubmit="handleSubmit(event)">
        <input type="text" id="userInput" placeholder="Ask a question about NVIDIA's annual reports or proxy statements (e.g., FY24 revenue, gross margin)..." autocomplete="off" />
        <button type="submit" class="send-btn" id="sendBtn">Send</button>
      </form>
    </div>
  </div>

  <script>
    function toggleSource(id) {
      const el = document.getElementById(id);
      if (el) {
        el.style.display = (el.style.display === 'block') ? 'none' : 'block';
      }
    }

    function askQuestion(q) {
      document.getElementById('userInput').value = q;
      document.getElementById('chatForm').dispatchEvent(new Event('submit'));
    }

    async function handleSubmit(e) {
      e.preventDefault();
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;

      input.value = '';
      const sendBtn = document.getElementById('sendBtn');
      sendBtn.disabled = true;

      const box = document.getElementById('messagesBox');

      // Append User Message
      const userRow = document.createElement('div');
      userRow.className = 'msg-row msg-user';
      userRow.innerHTML = `<div class="bubble bubble-user">${escapeHtml(text)}</div>`;
      box.appendChild(userRow);

      // Append Loading Bot Message
      const botRow = document.createElement('div');
      botRow.className = 'msg-row msg-bot';
      const botId = 'bot-' + Date.now();
      botRow.innerHTML = `<div class="bubble bubble-bot" id="${botId}"><span style="color:var(--text-muted)">Searching official NVIDIA filings and retrieving citations...</span></div>`;
      box.appendChild(botRow);
      box.scrollTop = box.scrollHeight;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: text })
        });
        const data = await res.json();

        let sourcesHtml = '';
        if (data.sources && data.sources.length > 0) {
          sourcesHtml = `<div class="sources-container"><div class="sources-title"><span>🔍 Verified Evidence Citations (${data.sources.length} Retrieved Chunks):</span></div>`;
          data.sources.forEach((s, idx) => {
            const sid = 'src-' + Date.now() + '-' + idx;
            sourcesHtml += `
              <div class="source-item">
                <div class="source-header" onclick="toggleSource('${sid}')">
                  <span>📄 <strong>${escapeHtml(s.citation)}</strong></span>
                  <span class="source-score">Similarity: ${(s.score * 100).toFixed(1)}% ▼</span>
                </div>
                <div class="source-body" id="${sid}">
                  <div><strong>Document:</strong> ${escapeHtml(s.title)} (FY${s.year}) | <strong>Page:</strong> ${s.page_number}</div>
                  <div style="margin-top:6px;">${escapeHtml(s.text_snippet)}</div>
                </div>
              </div>`;
          });
          sourcesHtml += '</div>';
        }

        let abstentionBanner = '';
        if (data.abstained) {
          abstentionBanner = `<div class="abstention-banner">⚠️ Appropriate Abstention Triggered: Insufficient evidence in ingested documents.</div>`;
        }

        document.getElementById(botId).innerHTML = `
          ${abstentionBanner}
          <div>${formatAnswer(data.answer)}</div>
          ${sourcesHtml}
        `;
      } catch (err) {
        document.getElementById(botId).innerHTML = `<span style="color:#ff5555">Error querying chatbot: ${escapeHtml(err.message)}</span>`;
      } finally {
        sendBtn.disabled = false;
        box.scrollTop = box.scrollHeight;
      }
    }

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function formatAnswer(text) {
      if (!text) return '';
      let formatted = escapeHtml(text);
      // Highlight inline citations like [NVIDIA ... Page ...]
      formatted = formatted.replace(/\\[([^\\]]+(?:Page|Report|Proxy)[^\\]]*)\\]/g, '<span style="background:rgba(118,185,0,0.18); color:#8ae000; padding:2px 6px; border-radius:4px; font-weight:600; font-size:12.5px;">[$1]</span>');
      return formatted.replace(/\\n\\n/g, '<br><br>').replace(/\\n/g, '<br>');
    }
  </script>
</body>
</html>
"""

class RAGRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "vector_store_loaded": engine.is_loaded}).encode("utf-8"))
        elif self.path == "/api/documents":
            from src.ingest import DOC_METADATA
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(DOC_METADATA).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
                question = (data.get("question") or data.get("query") or "").strip()
                if not question:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Empty question"}).encode("utf-8"))
                    return

                res = engine.query(question, top_k=5)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), RAGRequestHandler)
    print("=" * 70)
    print(f"NVIDIA RAG Chatbot Web Server active at: http://localhost:{port}")
    print(f"Health endpoint: http://localhost:{port}/api/health")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port=port)

