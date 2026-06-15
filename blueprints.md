# Local Modular AI Agent - Hybrid Architecture Blueprint (Python + Rust/Tauri)

## 1. System Architecture & Tech Stack

This project utilizes a **Hybrid Client-Server Architecture**. The heavy machine learning and system automation tasks run in a hidden Python background process. The UI, desktop window management, and avatar rendering are handled by a highly optimized Rust/Tauri application. They communicate in real-time via WebSockets.

### The Backend (The Brain)
* **Language:** Python 3.11
* **API/Routing:** `FastAPI` + `websockets`
* **Concurrency:** `asyncio` + `APScheduler` (for background tasks)
* **LLM & Vision Inference:** `Ollama` (Local hosting, JSON schema, streaming)
* **TTS (Text-to-Speech):** `Qwen3-TTS` (via `qwen-tts` with `flash-attn`)
* **STT (Speech-to-Text):** `Faster-Whisper` (via `CTranslate2`)
* **Memory Engine:** `ChromaDB` (Embedded) + `sentence-transformers`
* **OS Tools:** `pyautogui`, `Pillow`, `PyMuPDF`, `httpx` + `BeautifulSoup4`

### The Frontend (The Body)
* **Framework:** Tauri v2 (Rust Backend + Web Frontend)
* **UI Framework:** React + TypeScript + TailwindCSS
* **Avatar Engine:** Official Live2D Cubism Web SDK
* **OS Integration:** Native Rust window management (Transparency, Always-on-top, Global Hooks)

---

## 2. Directory Structure

```text
local-ai-assistant/
│
├── backend/                            # Python 3.11 Environment
│   ├── .venv/                          
│   ├── models/                         # Local offline weights cache
│   ├── storage/                        # SQLite DBs & ChromaDB chunks
│   │   └── cloned_voices/              # Precompiled Qwen3 .pt tensors
│   │
│   ├── skills/                         # Modular tool scripts
│   │   ├── __init__.py
│   │   ├── audio_pipeline.py           # WebRTCVAD -> Faster-Whisper
│   │   ├── system_nav.py               # pathlib OS traversal
│   │   ├── web_search.py               # SearXNG/DDG scraper
│   │   └── vision_capture.py           # Pillow multi-monitor capture
│   │
│   ├── src/                            # Core logic
│   │   ├── __init__.py
│   │   ├── orchestrator.py             # FastAPI WebSocket router
│   │   ├── memory_engine.py            # RAG & ChromaDB logic
│   │   ├── personality.py              # System prompts & JSON configs
│   │   └── tts_engine.py               # Qwen3-TTS inference loop
│   │
│   ├── requirements.txt
│   └── main.py                         # Python entry point
│
├── frontend/                           # Tauri + React GUI
│   ├── src-tauri/                      # Rust window & OS logic
│   │   ├── src/main.rs                 
│   │   ├── tauri.conf.json             # Window config (transparent, borderless)
│   │   └── Cargo.toml
│   │
│   ├── src/                            # React UI 
│   │   ├── components/                 # Chat UI, Settings, Live2D Canvas
│   │   ├── hooks/                      # useWebSocket, useAudioSync
│   │   └── assets/                     # Live2D Cubism models
│   │
│   ├── package.json
│   └── index.html
│
├── .gitignore
└── README.md
```

---

## 3. Implementation Phases

### Phase 1: Python Backend Foundation
1. **Initialize `backend/main.py`:** Create a FastAPI app exposing a WebSocket endpoint (`ws://127.0.0.1:8000/ws`).
2. **LLM Connection:** Implement `asyncio` wrappers to stream requests to the local Ollama instance.
3. **Context Management:** Build a sliding-window queue (e.g., `collections.deque`) to store active chat history.

### Phase 2: Frontend Desktop Shell
1. **Initialize Tauri:** Scaffold the `frontend/` directory using `create-tauri-app` (React/TS).
2. **Window Configuration:** Edit `tauri.conf.json` to enable `transparent: true`, `decorations: false`, and `alwaysOnTop: true`.
3. **WebSocket Client:** Build a React hook to connect to the Python backend and pass JSON messages back and forth.
4. **Global Shortcut:** Use Tauri's Rust API to register a system-wide hotkey (e.g., `Ctrl+Space`) to toggle window visibility.

### Phase 3: Audio & Voice Cloning
1. **TTS Engine (`backend/src/tts_engine.py`):** Write the compiler to convert a user `.wav` into a Qwen3-TTS `.pt` profile. Setup FlashAttention-2 streaming.
2. **STT Engine (`backend/skills/audio_pipeline.py`):** Implement `WebRTCVAD` to detect user speech, passing the audio buffer to `Faster-Whisper`.
3. **Frontend Playback:** Stream the raw audio bytes over the WebSocket to the React app for immediate playback via the Web Audio API.

### Phase 4: Long-Term Memory
1. **ChromaDB Setup:** Embed user-agent interactions using `all-MiniLM-L6-v2`.
2. **Background Summarization:** Use `APScheduler` to trigger a routine that condenses old chat logs into bulleted vector memories.
3. **Pre-generation Retrieval:** Inject top-3 relevant vector matches into the Ollama system prompt before generation.

### Phase 5: Live2D & Companion Mode
1. **Live2D SDK:** Mount the Cubism Web SDK to a `<canvas>` in React.
2. **Lip-Sync:** Calculate RMS volume from the incoming audio WebSocket stream and map it to `ParamMouthOpenY`.
3. **Eye Tracking:** Poll mouse coordinates via Tauri and normalize them to `ParamEyeBallX/Y`.
4. **Idle Detection:** Use Python's `pyautogui` to check for 5+ minutes of inactivity. Trigger a silent screen capture, pass it to the Vision LLM, and push a proactive dialogue event to the frontend.
