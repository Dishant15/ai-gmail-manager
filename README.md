# 📬 RAG Email Agent

An autonomous Gmail reply agent that uses a **local LLM** with **Retrieval-Augmented Generation (RAG)** to automatically draft and send professional email replies — powered by LangChain, ChromaDB, FastAPI, and your choice of LLM provider.

```
Gmail Inbox ──► FastAPI Poller ──► RAG Retrieval (ChromaDB)
                                          │
                             ┌────────────┴────────────┐
                             │                         │
                        Ollama Server           HuggingFace
                      (qwen2.5, mistral…)    (Mistral, Llama…)
                             │                         │
                             └────────────┬────────────┘
                                          │
                               Auto-Reply via Gmail API
```

---

## 🏗️ Architecture

```
rag-email-agent/
├── main.py                    # Uvicorn entry point
├── setup_gmail.py             # One-time Gmail OAuth2 setup
├── cleanup_mistral.py         # Remove cached HuggingFace model weights
├── requirements.txt
├── .env.example               # Copy to .env and configure
├── credentials.json           # Gmail OAuth (you provide this)
├── uploads/                   # Uploaded PDF knowledge base files
├── chroma_db/                 # ChromaDB vector store (auto-created)
│   ├── doc_metadata.json      # Document index metadata
│   ├── agent_config.json      # Persisted agent configuration
│   └── reply_log.json         # History of all replies
└── app/
    ├── config.py              # Pydantic settings + all data models
    ├── gmail_service.py       # Gmail OAuth2 + fetch/send emails
    ├── rag_engine.py          # PDF ingestion + ChromaDB + retrieval
    ├── agent.py               # Orchestrator + APScheduler polling
    ├── api.py                 # All FastAPI routes
    └── llm_services/          # Pluggable LLM provider system
        ├── __init__.py        # Provider router — reads LLM_PROVIDER from .env
        ├── base.py            # Shared prompt builder (used by all providers)
        ├── ollama_service.py  # Ollama provider (recommended)
        └── huggingface_service.py  # HuggingFace provider (CUDA machines)
```

### LLM provider system

The `llm_services/` package lets you switch LLM backends with a single env var — no code changes needed. The rest of the app (`agent.py`) always imports from `app.llm_services` and is completely unaware of which provider is active.

```
agent.py
   │
   └── from app.llm_services import generate_reply
              │
              ▼
        __init__.py  reads LLM_PROVIDER
              │
       ┌──────┴──────┐
       ▼             ▼
 ollama_service  huggingface_service
       │             │
  Ollama API    HuggingFace
  (external     Transformers
   process)     (in-process)
       │             │
       └──────┬──────┘
              ▼
        base.py  (shared prompt builder)
```

---

## ⚙️ Requirements

| Component | Ollama provider | HuggingFace provider |
|-----------|----------------|---------------------|
| Python | 3.10+ | 3.10+ |
| RAM | 8 GB+ (model runs outside Python) | 16 GB minimum, 32 GB recommended |
| Disk | Space for Ollama models (~4–8 GB per model) | ~20 GB (model weights) |
| GPU | Handled automatically by Ollama | NVIDIA CUDA optional |
| OS | Linux / macOS / Windows | Linux / macOS / Windows (WSL2) |
| Extra | Ollama installed | — |

> **Apple Silicon (M1/M2/M3/M4):** Use the **Ollama provider**. HuggingFace causes MPS buffer errors and CPU inference is very slow for 7B models. Ollama handles Apple Silicon natively and runs Qwen 2.5 at ~4–8 tokens/sec.

---

## 🦙 Running Ollama (recommended)

Ollama is a local model server that runs as a background process on your machine. It handles all model loading, memory management, and Apple Silicon optimisation automatically — you just point the agent at it.

### Install Ollama

Download and install from [https://ollama.com/download](https://ollama.com/download).

After installation, Ollama runs automatically as a background service. No manual starting needed — it launches at login.

### Pull a model

```bash
# Recommended for Apple Silicon M-series (fast, high quality)
ollama pull qwen2.5:7b

# Higher quality — needs ~16 GB free RAM
ollama pull qwen2.5:14b

# Alternative if you prefer Mistral
ollama pull mistral:7b

# Fastest option, lower quality — good for testing
ollama pull llama3.2:3b
```

### Verify Ollama is running

```bash
# List all pulled models
ollama list

# Test a model directly in terminal
ollama run qwen2.5:7b "Say hello"

# Check the API is reachable
curl http://localhost:11434
# Should return: Ollama is running
```

### Managing Ollama

```bash
# Check status
ollama list

# Remove a model (frees disk space)
ollama rm qwen2.5:7b

# See what is currently loaded in memory
ollama ps

# Stop a running model (free RAM)
ollama stop qwen2.5:7b

# Start Ollama manually if it stopped
ollama serve
```

### Ollama memory behaviour

By default Ollama keeps a model loaded in RAM for 5 minutes after the last request, then unloads it. This is controlled by `OLLAMA_KEEP_ALIVE` in `.env`:

```dotenv
OLLAMA_KEEP_ALIVE=5m    # unload after 5 min idle (default — saves RAM)
OLLAMA_KEEP_ALIVE=-1    # keep loaded forever (faster responses)
OLLAMA_KEEP_ALIVE=0     # unload immediately after each request
```

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

> **Apple Silicon Mac:** If `sentencepiece` fails to build, install the missing build tools first:
> ```bash
> brew install cmake pkg-config
> pip install -r requirements.txt
> ```

> **NVIDIA GPU (HuggingFace provider):** Install CUDA-enabled PyTorch first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Configure environment

```bash
cp .env.example .env
```

**For Ollama (recommended):**
```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
GMAIL_ADDRESS=you@gmail.com
EMAIL_POLL_INTERVAL=60
```

**For HuggingFace (NVIDIA GPU machines):**
```dotenv
LLM_PROVIDER=huggingface
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
HF_DEVICE=cuda
HF_LOAD_IN_4BIT=true
GMAIL_ADDRESS=you@gmail.com
```

### 5. Set up Gmail OAuth2

#### 5a. Create a Google Cloud project
1. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Navigate to **APIs & Services → Library**
4. Search for **Gmail API** → click **Enable**

#### 5b. Configure Branding
1. Go to **APIs & Services → OAuth consent screen** (may appear as **Google Auth Platform** in newer UI)
2. Click **"Branding"** in the left sidebar
3. Fill in the required fields:
   - App name: `RAG Email Agent`
   - User support email: your Gmail address
   - Developer contact email: your Gmail address
4. Click **Save**

#### 5c. Configure Audience & add yourself as a test user
1. Click **"Audience"** in the left sidebar
2. Make sure the publishing status is set to **"Testing"** — this is correct for personal use
3. Under **"Test users"** click **"+ Add users"**
4. Enter your own Gmail address and click **Save**

> **Why Testing mode?** In Testing mode only Gmail addresses you explicitly add as test users can authenticate. This is exactly what you want for a personal tool. Google requires a formal security audit to publish apps that access Gmail — for personal use Testing mode works permanently with no limitations.

#### 5d. Create OAuth credentials
1. Click **"Clients"** in the left sidebar (or **Credentials → Create Credentials → OAuth client ID**)
2. Click **"+ Create Client"**
3. Application type: **Desktop app** ← must be Desktop app, not Web application
4. Name: `RAG Email Agent`
5. Click **Create** → **Download JSON**
6. Rename the downloaded file to `credentials.json` and place it in the project root

> **Verify your credentials.json** — open it and confirm it starts with `"installed": {`. If it starts with `"web": {` the wrong application type was selected — delete and recreate as Desktop app.

#### 5e. Run the OAuth setup
```bash
python setup_gmail.py
```
A browser window will open. You may see an "unverified app" warning — click **Advanced → Go to RAG Email Agent (unsafe)** to proceed. Sign in with the Gmail address you added as a test user. A `token.json` is saved automatically.

---

#### 🔮 Future: what changes if you publish this app?

If you ever want other people's Gmail accounts to use this agent (not just your own), you would need to:

1. Go to **Audience** and click **"Publish app"** to switch from Testing to Production
2. Add the three Gmail OAuth scopes to the **Scopes** section and request verification
3. Submit the app for **Google's OAuth verification review** — this involves providing a privacy policy URL, a demo video, and waiting for Google's security team to audit the app (typically 1–4 weeks)
4. Until verification is approved, external users will see an "unverified app" warning

For personal use on your own Gmail account, none of this is needed — **Testing mode is permanent and has no expiry or limitations for the email address you added as a test user.**

### 6. Start the application

```bash
python main.py
```

The server starts at **http://localhost:8000**

- Interactive API docs: **http://localhost:8000/docs**
- OpenAPI schema: **http://localhost:8000/openapi.json**

> **Ollama provider:** Make sure Ollama is running and your model is pulled before starting. The agent will log a warning if the model is not found.

> **HuggingFace provider:** The model (~14 GB) downloads from HuggingFace Hub on first run and is cached locally for subsequent starts.

---

## 🔀 Switching LLM Providers

Change one line in `.env` and restart — no code changes needed:

```dotenv
LLM_PROVIDER=ollama       # use Ollama server (recommended)
LLM_PROVIDER=huggingface  # use HuggingFace Transformers
```

### Provider comparison

| | Ollama | HuggingFace |
|---|---|---|
| Apple Silicon | ✅ Native MPS support | ❌ OOM errors on 7B+ models |
| NVIDIA GPU | ✅ Supported | ✅ With 4-bit quantization |
| CPU fallback | ✅ Fast | ⚠️ Very slow on 7B models |
| Memory usage | Low (outside Python) | High (loaded in-process) |
| Speed (M4 Pro) | ~4–8 tok/s | ~1–3 tok/s (CPU only) |
| Model management | `ollama pull / rm` | HuggingFace Hub cache |
| Setup complexity | Install Ollama app | pip install only |

---

## 📡 API Reference

Full API documentation is in [`api_docs.md`](./api_docs.md).

### Quick reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/polling/status` | Agent status, interval, document count |
| `POST` | `/polling/trigger` | Manually run one inbox check |
| `POST` | `/polling/start` | Resume scheduled polling |
| `POST` | `/polling/stop` | Pause scheduled polling |
| `GET` | `/agent/config` | View current agent config |
| `PUT` | `/agent/config` | Update system prompt, signature, auto-reply |
| `POST` | `/agent/config/reset` | Reset config to defaults |
| `GET` | `/knowledge/documents` | List indexed PDFs |
| `POST` | `/knowledge/upload` | Upload + index a PDF |
| `DELETE` | `/knowledge/documents/{filename}` | Remove a PDF |
| `GET` | `/logs` | View reply history |

---

## 🛠️ Customisation

### Changing the Ollama model

```dotenv
OLLAMA_MODEL=qwen2.5:14b
```
Then: `ollama pull qwen2.5:14b` and restart the app.

### Changing the HuggingFace model

```dotenv
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
```
The new model downloads automatically on next start.

### Disabling auto-reply (draft mode)

```bash
curl -X PUT http://localhost:8000/agent/config \
  -H "Content-Type: application/json" \
  -d '{"auto_reply_enabled": false}'
```
The agent processes emails and logs what it would send, without actually sending.

### Adjusting RAG chunk size

```dotenv
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=150
RAG_TOP_K=3
```
Re-upload your PDFs after changing these — old chunks are replaced automatically.

---

## 🧹 Cleaning Up Model Files

If you used the HuggingFace provider and want to reclaim disk space, use the included cleanup script.

### Where HuggingFace model files live

| OS | Cache location |
|----|---------------|
| Linux / macOS | `~/.cache/huggingface/hub/` |
| Windows | `C:\Users\<you>\.cache\huggingface\hub\` |

### Running the cleanup script

```bash
# Preview what will be deleted — nothing is touched
python cleanup_mistral.py --dry-run

# Interactive mode — shows sizes, asks for confirmation
python cleanup_mistral.py

# Silent deletion, no prompt
python cleanup_mistral.py --force
```

### What the script deletes vs. preserves

| | Items |
|---|---|
| **Deleted** | HuggingFace model weights, embedding model cache, Torch kernel cache |
| **Never touched** | `./chroma_db/`, `./uploads/`, `credentials.json`, `token.json` |

> **Ollama models** are not affected by this script. To remove an Ollama model use `ollama rm <model>`.

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
- Both providers run entirely locally. No email content or documents leave your machine.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Ollama` | Open the Ollama app or run `ollama serve` in terminal |
| `Model not found in Ollama` | Run `ollama pull qwen2.5:7b` (or your configured model) |
| `Invalid buffer size` / MPS OOM | Switch to `LLM_PROVIDER=ollama` — HuggingFace is unstable on Apple Silicon |
| `sentencepiece` build fails | Run `brew install cmake pkg-config` then retry `pip install -r requirements.txt` |
| `credentials.json not found` | Follow step 5 — download from Google Cloud Console |
| `invalid_grant` OAuth error | Delete `token.json` and re-run `python setup_gmail.py` |
| `CUDA out of memory` | Set `HF_LOAD_IN_4BIT=true` or switch to `LLM_PROVIDER=ollama` |
| No emails being processed | Check `GET /polling/status` and `POST /polling/trigger` |
| Model download is slow | First run only — HuggingFace models are cached locally after download |