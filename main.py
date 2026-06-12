import os
import json
import time
import re
import warnings
from typing import List, Dict
from src.brain import AIBrain
from src.memory import LongTermMemory
from src.tts import TTSManager
import ollama

# Suppress Hugging Face/PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

def format_history_for_summary(history: List[Dict[str, str]]) -> str:
    """Formats the raw history list into a readable string for the summarizer LLM."""
    formatted = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Kiri"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def parse_json_list(text: str) -> List[str]:
    """Robustly parses a JSON list of strings from LLM output."""
    text = text.strip()
    # Remove markdown code blocks if present
    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try finding the brackets [ ]
        match_brackets = re.search(r'(\[.*?\])', text, re.DOTALL)
        if match_brackets:
            json_str = match_brackets.group(1)
        else:
            json_str = text
            
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
    except Exception:
        # Fallback to line splitting
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            line = re.sub(r'^[\-\*\+\d\.]+\s*', '', line)
            line = line.strip('"\',[]')
            if line:
                lines.append(line)
        return lines
    return []

def flush_input():
    """Flushes the input buffer to discard any keystrokes pressed while Kiri was generating or speaking."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        try:
            import sys
            import termios
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        except Exception:
            pass

def run_chat_generation(brain, memory, tts, user_input, conversation_history, injected_file_context=""):
    # Retrieve context from long-term memory
    print(f"\n[{brain.name} Memory] Searching past memories...")
    memories = memory.retrieve_context(user_input, limit=5)
    
    # Retrieve context from temporary folder RAG
    temp_memories = memory.retrieve_temp_context(user_input, limit=5)
    
    # Build message history for the model
    temp_history = conversation_history.copy()
    context_msgs = []
    
    if memories:
        context_str = "\n".join([f"- {m['summary']}" for m in memories])
        context_msgs.append(f"Relevant context from past conversations with the user:\n{context_str}")
        print(f"[{brain.name} Memory] Retrieved {len(memories)} relevant context fragment(s).")
        
    if temp_memories:
        temp_str = "\n".join([f"- [From {m['metadata']['source']}]: {m['summary']}" for m in temp_memories])
        context_msgs.append(f"Relevant context from active folder/files:\n{temp_str}")
        print(f"[{brain.name} Memory] Retrieved {len(temp_memories)} relevant active folder context chunk(s).")

    if injected_file_context:
        context_msgs.append(injected_file_context)
        
    if context_msgs:
        context_msg_content = "\n\n".join(context_msgs) + (
            "\n\nUse this context only if relevant to answer the user's input. "
            "Do not mention that you retrieved this from memory or loaded files unless asked."
        )
        temp_history.append({"role": "system", "content": context_msg_content})
    else:
        print(f"[{brain.name} Memory] No relevant memories or active folder contexts found.")
        
    # Append current user prompt
    temp_history.append({"role": "user", "content": user_input})
    
    # Generate and stream response
    print(f"\n{brain.name}: ", end="", flush=True)
    response_chunks = []
    try:
        for chunk in brain.generate_response(temp_history, stream=True):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)
        
        print() # End the line after stream complete
        
        # Clean up markdown formatting characters and enqueue full response block
        full_response = "".join(response_chunks).strip()
        if full_response:
            clean_text = re.sub(r'[*_`#\-+>]', '', full_response)
            clean_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_text)
            clean_text = clean_text.strip()
            if clean_text:
                tts.enqueue_speech(clean_text)
                
                # Block user input and show a dynamic indicator until Kiri finishes speaking
                spinner = ["|", "/", "-", "\\"]
                idx = 0
                while True:
                    status = tts.get_status()
                    if status == "idle":
                        break
                    elif status == "preparing":
                        symbol = spinner[idx % len(spinner)]
                        print(f"\r* {symbol} Kiri is preparing to speak... *", end="", flush=True)
                    elif status == "speaking":
                        symbol = spinner[idx % len(spinner)]
                        print(f"\r* {symbol} Kiri is speaking... *          ", end="", flush=True)
                    idx += 1
                    time.sleep(0.1)
                # Clear the status line completely
                print("\r" + " " * 50 + "\r", end="", flush=True)
                
        # Append actual message pair to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": full_response})
        print() # Padding line
    except Exception as e:
        print(f"\nError generating response: {e}")

def run_vision_analysis(brain, memory, tts, query, conversation_history):
    print(f"\n[{brain.name} Vision] Initializing screen capture...")
    brain.ensure_vision_model_available()
    
    try:
        from src.vision import capture_screen
        screenshot_path = "memory/screenshot.png"
        print(f"[{brain.name} Vision] Capturing primary screen...")
        capture_screen(screenshot_path)
        
        print(f"[{brain.name} Vision] Analyzing screen contents (private request)...")
        vision_prompt = (
            "Describe what is currently visible on this computer screen in detail: open windows, active applications, code, text, graphics, or activities."
        )
        # Call model privately to get visual context description
        vision_description = brain.generate_vision_response(vision_prompt, screenshot_path, stream=False)
        
        injected_vision_context = f"\n[Referenced Active Screen Contents]\nDescription of what is visible on the user's screen:\n{vision_description}\n"
        
        # Feed the description context to the main text LLM so Kiri responds in-character
        print(f"[{brain.name} Vision] Generating in-character visual response...")
        run_chat_generation(brain, memory, tts, query, conversation_history, injected_vision_context)
    except Exception as e:
        print(f"\n[{brain.name} Vision] Error processing screenshot: {e}")

def main():
    # Ensure ingestion folders exist
    os.makedirs("input_files", exist_ok=True)
    os.makedirs("archive_files", exist_ok=True)
    
    while True:
        print("==================================================")
        print("      Local AI VTuber Assistant (Kiri Brain)      ")
        print("==================================================")
        print("\nPlease select an option:")
        print("  [1] Start Chatting (GUI Mode - Default)")
        print("  [2] Start Chatting (Terminal Mode)")
        print("  [3] Recompile Voice Profile")
        print("  [4] Clear Long-Term Memory")
        print("  [5] Exit")
        
        try:
            flush_input()
            choice = input("\nSelect option [1-5] (Enter to start chatting in GUI mode): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return

        if choice == "" or choice == "1" or choice == "2":
            break
        elif choice == "3":
            print("\n[Menu] Recompiling speaker profile...")
            try:
                import prepare_voice
                prepare_voice.compile_voice()
            except Exception as e:
                print(f"[Menu] Error importing or compiling voice: {e}")
            print("\nPress Enter to return to menu...")
            input()
        elif choice == "4":
            print("\n[Menu] Clearing memory...")
            try:
                memory = LongTermMemory()
                memory.clear_memory()
                
                logs_dir = "memory/logs"
                if os.path.exists(logs_dir):
                    cleared_logs = 0
                    for filename in os.listdir(logs_dir):
                        file_path = os.path.join(logs_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                                cleared_logs += 1
                        except Exception as file_err:
                            print(f"[Menu] Failed to delete log {filename}: {file_err}")
                    if cleared_logs > 0:
                        print(f"[Memory System] Deleted {cleared_logs} session log file(s).")
            except Exception as e:
                print(f"[Menu] Error clearing memory: {e}")
            print("\nPress Enter to return to menu...")
            input()
        elif choice == "5" or choice.lower() in ["exit", "quit"]:
            print("\nExiting session. Goodbye!")
            return
        else:
            print("\n[Menu] Invalid option. Please select 1, 2, 3, 4, or 5.")
            time.sleep(1)

    # Initialize Brain, Memory, and TTS
    print("\n[System] Connecting to local Ollama service and initializing memory...")
    try:
        brain = AIBrain()
        memory = LongTermMemory()
        
        print("\n[TTS System] Initializing voice cloning model...")
        tts = TTSManager()
        tts.start()
        tts.ready_event.wait() # Block until model is fully loaded in memory
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    # Check if GUI or Terminal mode
    if choice == "" or choice == "1":
        print("\n[System] Starting Kiri in Desktop GUI Mode...")
        try:
            from src.gui import launch_gui
            launch_gui(brain, memory, tts)
        except Exception as e:
            print(f"[System] Failed to launch GUI: {e}")
            print("[System] Falling back to Terminal Mode...")
        else:
            return  # GUI execution finished cleanly, exit main()

    print(f"\n{brain.name} is online and ready! Type 'exit' or 'quit' to end the session.\n")
    
    conversation_history = []
    
    while True:
        try:
            flush_input()
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
            break

        # Intercept explicit commands
        if user_input.lower() == "/help":
            print("\n================ [Available Commands] ================")
            print("  /help                     - Show this help menu")
            print("  /see [optional query]     - Capture screen and analyze it")
            print("  /file <path>              - Inject file contents into chat context")
            print("  /dir <path>               - Load and index a directory temporarily")
            print("  /clear_dir                - Clear loaded directory context")
            print("  /exit or /quit            - Save memory logs and exit session")
            print("======================================================\n")
            continue
            
        elif user_input.lower().startswith(("/see", "/screen")):
            query_match = re.match(r'^/(?:see|screen)\s*(.*)$', user_input, re.IGNORECASE)
            query = query_match.group(1).strip() if query_match else ""
            if not query:
                query = "Describe this screenshot."
            run_vision_analysis(brain, memory, tts, query, conversation_history)
            continue
            
        elif user_input.lower().startswith("/file"):
            path_match = re.match(r'^/file\s*(.*)$', user_input, re.IGNORECASE)
            fpath = path_match.group(1).strip() if path_match else ""
            # Strip quotes if they were provided
            fpath = fpath.strip('"\'')
            if not fpath or not os.path.exists(fpath) or not os.path.isfile(fpath):
                msg = f"File not found or invalid path: {fpath}"
                print(f"\n{brain.name}: {msg}\n")
                tts.enqueue_speech(msg)
            else:
                from src.ingest import extract_text_from_file
                text = extract_text_from_file(fpath)
                filename = os.path.basename(fpath)
                injected_file_context = f"\n[Referenced File Context - {filename}]\nPath: {fpath}\nContent:\n```\n{text[:25000]}\n```\n"
                print(f"[{brain.name} Context] Injected file '{filename}' into context.")
                friendly_prompt = f"Please analyze and tell me about the file '{filename}'."
                run_chat_generation(brain, memory, tts, friendly_prompt, conversation_history, injected_file_context)
            continue
            
        elif user_input.lower().startswith("/dir"):
            path_match = re.match(r'^/dir\s*(.*)$', user_input, re.IGNORECASE)
            dpath = path_match.group(1).strip() if path_match else ""
            # Strip quotes if they were provided
            dpath = dpath.strip('"\'')
            if not dpath or not os.path.exists(dpath) or not os.path.isdir(dpath):
                msg = f"Directory not found or invalid path: {dpath}"
                print(f"\n{brain.name}: {msg}\n")
                tts.enqueue_speech(msg)
            else:
                print(f"\n[{brain.name} Ingest] Scanning and indexing directory '{dpath}'...")
                indexed_count = memory.set_temp_directory(dpath)
                confirm_msg = f"Successfully loaded directory '{os.path.basename(dpath)}' ({indexed_count} files indexed) into temporary context memory!"
                print(f"\n{brain.name}: {confirm_msg}\n")
                tts.enqueue_speech(confirm_msg)
            continue
            
        elif user_input.lower().startswith(("/clear_dir", "/clear_context")):
            memory.clear_temp_context()
            confirm_msg = "Cleared temporary context directory."
            print(f"\n{brain.name}: {confirm_msg}\n")
            tts.enqueue_speech(confirm_msg)
            continue

        # Implicit conversational intent analysis
        from src.intent import detect_vision_trigger, find_paths_in_prompt
        
        # 1. Implicit Vision Trigger Check
        if detect_vision_trigger(user_input):
            print(f"[{brain.name} Vision] Automatically captured screen for context...")
            run_vision_analysis(brain, memory, tts, user_input, conversation_history)
            continue
            
        # 2. File/Folder Mention Scanner
        detected_files, detected_dirs = find_paths_in_prompt(user_input)
        injected_file_context = ""
        
        # If directory path mentioned along with context setting keywords
        folder_keywords = ["folder", "directory", "context", "use", "set", "load", "reference", "look at"]
        has_folder_kw = any(kw in user_input.lower() for kw in folder_keywords)
        
        if detected_dirs and has_folder_kw:
            dpath = detected_dirs[0]
            print(f"\n[{brain.name} Ingest] Automatically loading directory '{dpath}'...")
            indexed_count = memory.set_temp_directory(dpath)
            confirm_msg = f"I've loaded directory '{os.path.basename(dpath)}' ({indexed_count} files) into temporary context memory!"
            print(f"\n{brain.name}: {confirm_msg}\n")
            tts.enqueue_speech(confirm_msg)
            
        if detected_files:
            from src.ingest import extract_text_from_file
            for fpath in detected_files[:3]:
                text = extract_text_from_file(fpath)
                if text.strip():
                    filename = os.path.basename(fpath)
                    print(f"[{brain.name} Context] Automatically loaded file '{filename}' into context.")
                    injected_file_context += f"\n[Referenced File Context - {filename}]\nPath: {fpath}\nContent:\n```\n{text[:25000]}\n```\n"

        # Regular Chat Generation (queries RAG over long-term & temp memories)
        run_chat_generation(brain, memory, tts, user_input, conversation_history, injected_file_context)


    # Handle session end, fact extraction, and long-term memory injection
    if conversation_history:
        print(f"\n[{brain.name} Memory] Extracting key facts and preferences from conversation...")
        conversation_text = format_history_for_summary(conversation_history)
        
        extraction_prompt = (
            "Analyze the following conversation between the User and the AI assistant (Kiri).\n"
            "Extract a list of discrete, key facts, preferences, user hobbies, details about the user, or tasks/reminders that the User shared.\n"
            "Rules:\n"
            "1. Each fact must be a short, standalone sentence (e.g., 'User has a cat named Whiskers', 'User prefers Python for coding').\n"
            "2. Keep them completely objective and factual. Do not include introductory phrases or chat details.\n"
            "3. Format the output as a simple JSON list of strings: [\"fact 1\", \"fact 2\", ...].\n"
            "4. If no specific facts or user preferences were shared, return an empty list: [].\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            "Output JSON only:"
        )
        
        try:
            # Generate facts list
            summary_response = ollama.chat(
                model=brain.model_name,
                messages=[{"role": "user", "content": extraction_prompt}],
                stream=False
            )
            raw_output = summary_response["message"]["content"].strip()
            facts = parse_json_list(raw_output)
            
            if facts:
                print(f"[{brain.name} Memory] Extracted {len(facts)} new fact(s):")
                history_json = json.dumps(conversation_history, ensure_ascii=False)
                for idx, fact in enumerate(facts):
                    print(f"  - {fact}")
                    memory.add_memory(fact, history_json)
            else:
                print(f"[{brain.name} Memory] No significant facts or user preferences extracted from this session.")
            
            # Save raw log file for reference
            logs_dir = "memory/logs"
            os.makedirs(logs_dir, exist_ok=True)
            log_filename = f"{logs_dir}/chat_{int(time.time())}.json"
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, indent=2, ensure_ascii=False)
            print(f"[{brain.name} Memory] Session logs saved to {log_filename}")
            
        except Exception as e:
            print(f"[{brain.name} Memory] Failed to save long-term memory: {e}")

    # Shut down TTS cleanly
    if 'tts' in locals():
        print(f"\n[{brain.name} Memory] Waiting for speech playback to finish...")
        tts.wait_until_done()
        tts.stop()

    print(f"[{brain.name} Memory] Offline.")

if __name__ == "__main__":
    main()
