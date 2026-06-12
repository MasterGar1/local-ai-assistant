# 🚀 Local AI VTuber Assistant: Complete Execution Plan

This document outlines a comprehensive, step-by-step plan to build your local AI assistant with a transparent VTuber interface, strictly utilizing local processing to run optimally on a 16GB VRAM GPU. 

---

## 🛠️ Proposed Tech Stack (16GB VRAM Optimized)

To fit everything in 16GB of VRAM while avoiding a webapp interface, we must carefully select lightweight but powerful components:

* **Core Language:** Python 3.10+
* **LLM Engine:** [Ollama](https://ollama.com/) (Extremely optimized for VRAM usage via llama.cpp).
    * *Text Model:* `Llama-3-8B-Instruct` or `Qwen-2.5-7B` (Quantized to 4-bit, takes ~5.5GB VRAM).
    * *Vision Model:* `Llava` or `llama3.2-vision` (Loaded on demand, ~5-6GB VRAM).
* **Text-to-Speech (Voice Cloning):** [Coqui XTTSv2](https://github.com/coqui-ai/TTS) or [CosyVoice](https://github.com/FunAudioLLM/CosyVoice). XTTSv2 requires only ~3GB VRAM and allows 3-second zero-shot voice cloning.
* **Long-Term Memory:** `ChromaDB` (Local Vector Database) + `sentence-transformers` (all-MiniLM-L6-v2) to embed and retrieve context without heavy memory overhead.
* **Desktop UI:** `PyQt6` or `PySide6`. Highly robust for desktop applications. Allows for frameless, background-transparent, always-on-top windows.
* **VTuber Rendering:** `PyQt6-WebEngine`. Python natively struggles to render Live2D or VRM models directly without heavy game engines. The clever workaround is to load a local, lightweight HTML/WebGL script (using `pixi-live2d-display` for 2D or `three-vrm` for 3D) inside a transparent PyQt widget.
* **Screenshot/Vision:** `mss` (ultra-fast cross-platform screenshotting) and `watchdog` (for folder monitoring).

---

## 🗺️ Step-by-Step Implementation Plan

### Step 1: Base Environment & Ollama Integration
**Goal:** Setup the core Python text-in/text-out pipeline.
1.  **Environment:** Create a Python virtual environment and install `ollama` Python client.
2.  **System Prompt Configuration:** Create a configuration file (JSON/YAML) defining the assistant's personality. This file will be injected as a system prompt.
3.  **Basic Chat Loop:** Write a Python script that takes user input from the terminal, appends the personality system prompt, sends it to the local Ollama instance, and streams the text response back.

### Step 2: Long-Term Memory System
**Goal:** Give the assistant the ability to remember past conversations.
1.  **Conversation Summarization:** At the end of a session (or after X messages), trigger a background prompt to Ollama: *"Summarize the following conversation concisely."*
2.  **Save to File:** Save the raw conversation and the summary to a local JSON file in a `/memory` directory.
3.  **Vector Database Integration:** Initialize `ChromaDB` locally. Embed the *summary* of the conversation using a lightweight embedding model (e.g., `sentence-transformers`).
4.  **Retrieval-Augmented Generation (RAG):** When the user starts a new conversation or asks a question, embed their prompt, query ChromaDB for the closest past conversations, and inject that context into the LLM prompt.

### Step 3: Local Text-to-Speech (Voice Cloning)
**Goal:** Make the assistant speak using a cloned voice.
1.  **Setup TTS:** Install `TTS` (Coqui) or `CosyVoice`. 
2.  **Voice Cloning Prep:** Provide a high-quality 10-second `.wav` audio clip of the character's voice.
3.  **Text Chunking:** LLM outputs text in blocks. Use sentence boundary detection to chunk the text (split by periods/newlines) so TTS can start generating audio *while* the LLM is still typing.
4.  **Audio Playback:** Use `sounddevice` or `pygame.mixer` to play the generated audio files synchronously in Python.

### Step 4: Vision & Folder Ingestion
**Goal:** Allow the assistant to "see" your screen and read local files.
1.  **Screen Capture:** Implement `mss` to capture the current screen content upon a specific user trigger.
2.  **Folder Ingestion:** Write a script that can read text files, PDFs (via `PyMuPDF`), and images in a specified user folder.
3.  **Multimodal Processing:** When vision is triggered, switch the Ollama call from the standard LLM to a vision model (e.g., `llava`). Pass the image array/file alongside the user's prompt. *Note: Ensure your code unloads the text model to free up VRAM for the vision model if needed.*

### Step 5: Desktop UI (Transparent & Always-on-Top)
**Goal:** Create the non-webapp, native OS desktop window.
1.  **PyQt6 Setup:** Create a `QMainWindow`.
2.  **Window Flags:** Apply the following Qt attributes to make the window transparent and always on top:
    * `Qt.FramelessWindowHint` (Removes borders and OS title bar)
    * `Qt.WindowStaysOnTopHint` (Keeps it above other windows)
    * `setAttribute(Qt.WA_TranslucentBackground)` (Makes the background invisible)
3.  **Draggable Logic:** Implement custom mouse events (`mousePressEvent`, `mouseMoveEvent`) so the user can click and drag the invisible window around their screen.
4.  **Chat UI Toggle:** Add a keyboard shortcut or a right-click context menu that pops open a text input box (`QLineEdit` + `QTextEdit`) attached to the bottom of the character.

### Step 6: VTuber Model Rendering & Lip Sync
**Goal:** Bring the character to life on the screen.
1.  **Local WebRenderer:** Inside the PyQt6 app, instantiate a `QWebEngineView`. Make its background transparent.
2.  **Load Model:** Point the WebEngine to a local `index.html` file that uses `three-vrm` (for 3D models) or Live2D WebGL SDK (for 2D models). This HTML simply renders the character on a clear canvas.
3.  **Lip Sync Integration:** * Extract audio amplitude (volume) dynamically during Python audio playback using `numpy`.
    * Pass this amplitude value continuously to the WebEngine via `QWebEnginePage.runJavaScript()`.
    * In the local HTML/JS script, map this amplitude to the VTuber model's blendshapes (specifically the `aa`, `ih`, `ou`, `ee`, `oh` mouth shapes or simple jaw open variables).

### Step 7: Asynchronous Pipeline Assembly
**Goal:** Stitch it all together without freezing the UI.
1.  **Thread Management:** If the LLM generation or TTS runs on the main UI thread, the VTuber model will freeze.
2.  **QThreads/Asyncio:** Wrap the `Ollama` calls, the `TTS` generation, and the `Audio Playback` into separate `QThread` workers.
3.  **Signals and Slots:** Use PyQt's Signal/Slot system to communicate between the background AI worker and the main UI (e.g., "AI is thinking...", "AI is speaking...").

---

## 🧠 Hardware & VRAM Management Strategy (16GB Limit)
* **Operating System:** Windows/Linux. 
* **Memory Budget:**
    * OS/Desktop: `~1.5GB`
    * Ollama (Llama-3 8B 4-bit): `~5.5GB`
    * TTS Model (XTTSv2): `~3.5GB`
    * ChromaDB / Embedding: `~1.0GB`
    * PyQt / VTuber WebGL Engine: `~1.0GB`
    * **Total:** `~12.5GB` (Leaves 3.5GB buffer for vision models or context windows).
* **Tip:** Set Ollama's `keep_alive` parameter optimally so it drops the model from memory when the system sits idle, preventing your computer from lagging during regular tasks.

---

## 🗓️ Recommended Execution Order for Development
1. Start with the **LLM + Personality + Memory** entirely in the terminal. Prove the brain works.
2. Add the **TTS + Voice Cloning** to the terminal script. Prove it can speak.
3. Build the **PyQt6 Transparent Window** with a static image first. Prove the UI behaves correctly.
4. Embed the **VTuber local web view** into the transparent window.
5. Connect the terminal brain to the UI brain and map the audio output to the VTuber's lip sync.
6. Finally, add the **Vision and Folder Ingestion** capabilities.
