# 📡 RAG Email Agent — API Documentation

Base URL: `http://localhost:8000`

Interactive docs (no curl needed): **http://localhost:8000/docs**

---

## Table of Contents

- [Health](#-health)
- [Polling — Inbox Monitoring](#-polling--inbox-monitoring)
- [Agent Configuration](#-agent-configuration)
- [Knowledge Base — PDF Management](#-knowledge-base--pdf-management)
- [Logs](#-logs)
- [Controlling Auto-Replies](#-controlling-auto-replies)

---

## 🟢 Health

### `GET /`
Check that the server is running.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "RAG Email Agent is running",
  "docs": "/docs"
}
```

---

### `GET /health`
Lightweight health check — use this for uptime monitoring.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Service is healthy"
}
```

---

## 📬 Polling — Inbox Monitoring

The agent automatically checks Gmail every 60 seconds (configurable via `EMAIL_POLL_INTERVAL` in `.env`). These endpoints let you inspect and control that scheduler.

---

### `GET /polling/status`
View the current state of the agent — whether polling is running, how often, and how many documents are indexed.

```bash
curl http://localhost:8000/polling/status
```

**Response:**
```json
{
  "running": true,
  "interval_seconds": 60,
  "gmail_address": "you@gmail.com",
  "auto_reply_enabled": true,
  "documents_indexed": 3
}
```

---

### `POST /polling/trigger`
Manually run one inbox check cycle immediately, without waiting for the next scheduled interval. Returns a list of emails processed in this cycle.

```bash
curl -X POST http://localhost:8000/polling/trigger
```

**Response:**
```json
[
  {
    "email_id": "18f3a2c...",
    "subject": "Question about your services",
    "sender": "customer@example.com",
    "reply_preview": "Thank you for reaching out. Based on our knowledge base...",
    "sent": true,
    "timestamp": "2025-04-30T10:22:00Z"
  }
]
```

Returns an empty array `[]` if there are no unread emails.

---

### `POST /polling/start`
Resume the scheduled inbox polling if it was paused.

```bash
curl -X POST http://localhost:8000/polling/start
```

**Response:**
```json
{
  "status": "ok",
  "message": "Polling started"
}
```

---

### `POST /polling/stop`
Pause the scheduled inbox polling. The app keeps running but the agent stops checking for new emails. **Note:** this does not persist across restarts — polling resumes automatically when the app is restarted.

```bash
curl -X POST http://localhost:8000/polling/stop
```

**Response:**
```json
{
  "status": "ok",
  "message": "Polling paused"
}
```

---

## ⚙️ Agent Configuration

These endpoints control the agent's behaviour — its persona, reply signature, whether it actually sends emails, and how long its replies can be. All settings are **persisted to disk** (`chroma_db/agent_config.json`) and survive app restarts.

---

### `GET /agent/config`
View the current agent configuration.

```bash
curl http://localhost:8000/agent/config
```

**Response:**
```json
{
  "system_prompt": "You are a helpful email assistant...",
  "reply_signature": "\n\nBest regards,\nYour AI Assistant",
  "auto_reply_enabled": true,
  "max_reply_tokens": 512
}
```

---

### `PUT /agent/config`
Update one or more configuration fields. All fields are optional — only the fields you include will be changed.

```bash
curl -X PUT http://localhost:8000/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a customer support agent for Acme Corp. Always be concise and offer a solution.",
    "reply_signature": "\n\nBest,\nSupport Team at Acme Corp",
    "auto_reply_enabled": true,
    "max_reply_tokens": 512
  }'
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `system_prompt` | string | Controls the agent's persona and behaviour. Injected into every Mistral prompt. |
| `reply_signature` | string | Appended to the end of every reply. Use `\n\n` for line breaks. |
| `auto_reply_enabled` | boolean | `true` = send replies automatically. `false` = draft mode, logs replies but does not send. |
| `max_reply_tokens` | integer | Maximum length of generated reply. Range: 64–2048. |

**Response:** the full updated config object (same shape as `GET /agent/config`).

---

### `POST /agent/config/reset`
Reset all agent configuration fields back to their defaults.

```bash
curl -X POST http://localhost:8000/agent/config/reset
```

**Defaults restored:**
```json
{
  "system_prompt": "You are a helpful email assistant. Use the provided context from the knowledge base to answer emails accurately and professionally. Keep replies concise and polite. If the context does not contain enough information to answer, say so honestly.",
  "reply_signature": "\n\nBest regards,\nYour AI Assistant",
  "auto_reply_enabled": true,
  "max_reply_tokens": 512
}
```

---

## 📚 Knowledge Base — PDF Management

These endpoints manage the PDFs the agent uses as its knowledge source. Uploaded files are split into chunks, embedded, and stored in ChromaDB. The agent retrieves the most relevant chunks for every incoming email.

---

### `GET /knowledge/documents`
List all currently indexed PDF documents.

```bash
curl http://localhost:8000/knowledge/documents
```

**Response:**
```json
[
  {
    "filename": "faq.pdf",
    "num_chunks": 47,
    "upload_time": "2025-04-30T10:00:00Z",
    "file_size_kb": 312.4
  },
  {
    "filename": "product_manual.pdf",
    "num_chunks": 112,
    "upload_time": "2025-04-30T11:30:00Z",
    "file_size_kb": 891.2
  }
]
```

---

### `POST /knowledge/upload`
Upload a PDF file and index it into the knowledge base. Re-uploading the same file (same content) is a no-op — it will be skipped automatically.

```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "num_chunks": 63,
  "upload_time": "2025-04-30T12:00:00Z",
  "file_size_kb": 450.1
}
```

**Errors:**
- `400` — file is not a PDF
- `500` — failed to parse or index the PDF

---

### `DELETE /knowledge/documents/{filename}`
Remove a document from the knowledge base. Deletes all its chunks from ChromaDB and removes the file from disk.

```bash
curl -X DELETE http://localhost:8000/knowledge/documents/faq.pdf
```

**Response:**
```json
{
  "status": "ok",
  "message": "Document 'faq.pdf' deleted."
}
```

**Error:**
- `404` — document not found in the index

---

## 📋 Logs

### `GET /logs`
Retrieve the history of emails the agent has processed and replied to (or drafted). Returns newest first.

```bash
# Latest 50 (default)
curl http://localhost:8000/logs

# Custom limit
curl http://localhost:8000/logs?limit=10
```

**Response:**
```json
[
  {
    "email_id": "18f3a2c...",
    "subject": "Question about your services",
    "sender": "customer@example.com",
    "reply_preview": "Thank you for reaching out. Based on our knowledge base...",
    "sent": true,
    "timestamp": "2025-04-30T10:22:00Z"
  }
]
```

**Fields:**

| Field | Description |
|-------|-------------|
| `email_id` | Gmail message ID |
| `subject` | Subject line of the original email |
| `sender` | Sender's email address |
| `reply_preview` | First 300 characters of the generated reply |
| `sent` | `true` if the reply was sent, `false` if drafted only |
| `timestamp` | When the reply was generated (UTC) |

**Query params:**

| Param | Default | Max | Description |
|-------|---------|-----|-------------|
| `limit` | `50` | `200` | Number of log entries to return |

---

## 🔁 Controlling Auto-Replies

### Start auto-replies
```bash
curl -X PUT http://localhost:8000/agent/config \
  -H "Content-Type: application/json" \
  -d '{"auto_reply_enabled": true}'
```

### Stop auto-replies (draft mode)
The agent still processes incoming emails and logs what it *would* have sent — it just doesn't send them. Useful for reviewing replies before going live.

```bash
curl -X PUT http://localhost:8000/agent/config \
  -H "Content-Type: application/json" \
  -d '{"auto_reply_enabled": false}'
```

### Pause inbox polling entirely
Stops the agent from checking the inbox at all. Does **not** persist across restarts.

```bash
curl -X POST http://localhost:8000/polling/stop
```

### Resume inbox polling
```bash
curl -X POST http://localhost:8000/polling/start
```

### Difference between draft mode and pausing polling

| | `auto_reply_enabled: false` | `POST /polling/stop` |
|---|---|---|
| Checks inbox | ✅ Yes | ❌ No |
| Generates reply | ✅ Yes | ❌ No |
| Sends reply | ❌ No | ❌ No |
| Logs the draft | ✅ Yes (visible in `/logs`) | ❌ No |
| Persists after restart | ✅ Yes | ❌ No |

Use `auto_reply_enabled: false` when you want to **review what the agent would send** without it actually sending.
Use `polling/stop` when you want to **pause the whole agent temporarily** while keeping the app running.

---

## 💡 Tips

- **No curl?** All endpoints are available in the interactive browser UI at `http://localhost:8000/docs` — click any endpoint → "Try it out" → "Execute".
- **All config changes are instant** — no restart needed.
- **Knowledge base changes are instant** — uploaded PDFs are immediately available for the next email the agent processes.
- **Logs are capped at 200 entries** — oldest entries are dropped automatically as new ones are added.
