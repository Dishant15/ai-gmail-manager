# 📬 RAG Email Agent

An autonomous Gmail reply agent that uses a **local Mistral 7B LLM** with **Retrieval-Augmented Generation (RAG)** to automatically draft and send professional email replies — powered by LangChain, ChromaDB, and FastAPI.

```
Gmail Inbox ──► FastAPI Poller ──► RAG Retrieval (ChromaDB)
                                         │
                                    Mistral 7B (local)
                                         │
                              Auto-Reply via Gmail API
```

---

## 🏗️ Architecture

```
rag-email-agent/
├── main.py                  # Uvicorn entry point
├── setup_gmail.py           # One-time OAuth2 setup
├── cleanup_mistral.py       # Remove cached model weights from disk
├── requirements.txt
├── .env.example             # Copy to .env and configure
├── credentials.json         # Gmail OAuth (you provide this)
├── uploads/                 # Uploaded PDF knowledge base files
├── chroma_db/               # ChromaDB vector store (auto-created)
│   ├── doc_metadata.json    # Document index metadata
│   ├── agent_config.json    # Persisted agent configuration
│   └── reply_log.json       # History of all replies
└── app/
    ├── config.py            # Pydantic settings + data models
    ├── gmail_service.py     # Gmail OAuth2 + fetch/send
    ├── rag_engine.py        # PDF ingestion + ChromaDB + retrieval
    ├── llm_service.py       # Local Mistral via HuggingFace
    └── agent.py             # Orchestrator + scheduler
    └── api.py               # All FastAPI routes
```

---

## ⚙️ Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.10+ |
| RAM | 16 GB (32 GB recommended for 4-bit model) |
| Disk | 20 GB free (model weights ~14 GB) |
| GPU | Optional — NVIDIA CUDA for acceleration |
| OS | Linux / macOS / Windows (WSL2) |

---

## 🚀 Setup & Installation

### 1. Clone / download the project

```bash
cd rag-email-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — the key values to set:

```dotenv
GMAIL_ADDRESS=you@gmail.com
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3   # or Mistral-7B-Instruct-v0.2
HF_DEVICE=auto          # auto = GPU if available, else CPU
HF_LOAD_IN_4BIT=true    # saves ~8 GB VRAM, requires CUDA
EMAIL_POLL_INTERVAL=60  # seconds between inbox checks
```

### 5. Set up Gmail OAuth2

#### 5a. Create a Google Cloud project
1. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Navigate to **APIs & Services → Library**
4. Search for **Gmail API** → click **Enable**

#### 5b. Create OAuth credentials
1. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
2. Application type: **Desktop app**
3. Click **Download JSON**
4. Rename the downloaded file to `credentials.json` and place it in the project root

#### 5c. Run the OAuth setup
```bash
python setup_gmail.py
```
A browser window will open. Sign in and grant access. A `token.json` is saved automatically.

### 6. Start the application

```bash
python main.py
```

The server starts at **http://localhost:8000**

- Interactive API docs: **http://localhost:8000/docs**
- OpenAPI schema: **http://localhost:8000/openapi.json**

> **First run:** The Mistral model (~14 GB) will be downloaded from HuggingFace Hub. This takes time depending on your internet connection. It is cached locally and reused on subsequent starts.

---

## 📡 API Reference

### Polling

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/polling/status` | Agent status, interval, document count |
| `POST` | `/polling/trigger` | Manually run one inbox check cycle |
| `POST` | `/polling/start` | Resume scheduled polling |
| `POST` | `/polling/stop` | Pause scheduled polling |

### Agent Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/agent/config` | View current agent config |
| `PUT` | `/agent/config` | Update system prompt, signature, auto-reply toggle |
| `POST` | `/agent/config/reset` | Reset to defaults |

**Example — update system prompt:**
```bash
curl -X PUT http://localhost:8000/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a customer support agent for Acme Corp. Be concise and always offer a solution.",
    "reply_signature": "\n\nBest,\nSupport Team",
    "auto_reply_enabled": true
  }'
```

### Knowledge Base

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/knowledge/documents` | List indexed PDFs |
| `POST` | `/knowledge/upload` | Upload + index a PDF |
| `DELETE` | `/knowledge/documents/{filename}` | Remove a PDF |

**Example — upload a PDF:**
```bash
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@/path/to/faq.pdf"
```

### Logs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/logs?limit=50` | View reply history |

---

## 🛠️ Customisation

### Changing the LLM

Edit `.env`:
```dotenv
# Smaller / faster
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2

# More capable (requires more VRAM)
HF_MODEL_ID=mistralai/Mixtral-8x7B-Instruct-v0.1
```

### Disabling auto-reply (draft mode)

```bash
curl -X PUT http://localhost:8000/agent/config \
  -H "Content-Type: application/json" \
  -d '{"auto_reply_enabled": false}'
```
The agent will process emails and log generated replies without sending them.

### Adjusting RAG chunk size

Edit `.env`:
```dotenv
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=150
RAG_TOP_K=3
```
Then re-upload your PDFs (old chunks are replaced automatically).

---

## 🧹 Cleaning Up Model Files

If you no longer need Mistral (e.g. switching to a different LLM), use the included cleanup script to reclaim the ~14 GB of disk space taken by locally cached model weights.

### Where the files live

HuggingFace never stores weights inside the project folder. Everything goes into a global cache:

| OS | Cache location |
|----|---------------|
| Linux / macOS | `~/.cache/huggingface/hub/` |
| Windows | `C:\Users\<you>\.cache\huggingface\hub\` |

Inside that folder, `mistralai/Mistral-7B-Instruct-v0.3` is stored as `models--mistralai--Mistral-7B-Instruct-v0.3` — a single directory holding all weight shards, tokenizer files, and config (~14 GB). The embedding model and PyTorch kernel cache add a smaller amount on top.

### Running the cleanup script

```bash
# 1. Preview what will be deleted — nothing is touched
python cleanup_mistral.py --dry-run

# 2. Interactive mode — shows sizes, asks for confirmation before deleting
python cleanup_mistral.py

# 3. Silent deletion, no confirmation prompt
python cleanup_mistral.py --force
```

### Example output

```
════════════════════════════════════════════════════════════
  Mistral & HuggingFace Cache Cleanup
════════════════════════════════════════════════════════════
  HuggingFace cache : /home/you/.cache/huggingface/hub
  Torch cache       : /home/you/.cache/torch

  Targets to clean up:
  Status       Size         Label
  ─────────────────────────────────────────────────────────────
  FOUND        13.84 GB     Model weights: mistralai/Mistral-7B-Instruct-v0.3
  not found    —            Model weights: mistralai/Mixtral-8x7B-Instruct-v0.1
  FOUND        88.3 MB      Model weights: sentence-transformers/all-MiniLM-L6-v2
  FOUND        142 MB       Torch cache (compiled kernels)
  ─────────────────────────────────────────────────────────────
  Total reclaimable:         14.07 GB

  Proceed? [y/N]
```

### What the script deletes vs. preserves

| | Items |
|---|---|
| **Deleted** | Mistral 7B weights (all v0.1/v0.2/v0.3 variants found), embedding model weights, HuggingFace accelerate cache, Torch compiled kernel cache |
| **Never touched** | `./chroma_db/` (your vector store), `./uploads/` (your PDFs), `credentials.json` / `token.json`, any other Python packages or projects |

> **Switching models:** After cleanup, update `HF_MODEL_ID` in `.env` to any other HuggingFace model ID. The new model will be downloaded automatically on the next application start.

---

## 🔒 Security Notes

- `credentials.json` and `token.json` grant access to your Gmail account — **never commit these to version control**.
- Add both to `.gitignore`:
  ```
  credentials.json
  token.json
  .env
  chroma_db/
  uploads/
  ```
- The application runs entirely locally. No email content or documents are sent to external services (only HuggingFace Hub for model download).

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `credentials.json not found` | Follow step 5 — download from Google Cloud Console |
| `CUDA out of memory` | Set `HF_LOAD_IN_4BIT=true` or `HF_DEVICE=cpu` |
| Model download is slow | Be patient on first run — it is cached afterward |
| `invalid_grant` OAuth error | Delete `token.json` and re-run `setup_gmail.py` |
| No emails being processed | Check `GET /polling/status` and `POST /polling/trigger` |
