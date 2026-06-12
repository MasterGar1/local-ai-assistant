import os
import sys
import queue
import threading
import warnings
import torch
import sounddevice as sd
import numpy as np
from contextlib import contextmanager
from huggingface_hub.utils import disable_progress_bars
from transformers import logging as transformers_logging

# Suppress Hugging Face/PyTorch warnings and progress bars to keep terminal clean
disable_progress_bars()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
transformers_logging.set_verbosity_error()

@contextmanager
def silence_stdout_stderr():
    """Context manager to redirect stdout/stderr to devnull, with a safe fallback if OS-level dup2 is not supported."""
    devnull_fd = None
    saved_stdout_fd = None
    saved_stderr_fd = None
    os_redirect_success = False
    
    try:
        # Try OS-level descriptor redirection
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os_redirect_success = True
    except Exception:
        # Fallback if OS-level dup/dup2 is not supported in the current terminal shell/sandbox
        pass
        
    # Apply python-level stream redirection (always safe and cross-platform)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        yield
    finally:
        # Restore python-level streams
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Restore OS-level descriptors if successfully redirected
        if os_redirect_success:
            try:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)
                os.close(devnull_fd)
            except Exception:
                pass
        elif devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except Exception:
                pass

class TTSManager(threading.Thread):
    """Manages the background queue-based Qwen3-TTS generation and audio playback."""
    
    def __init__(self, voice_dir: str = "config/voice", model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"):
        super().__init__(daemon=True)
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.voice_dir = voice_dir
        self.model_id = model_id
        self.running = True
        self.ready_event = threading.Event()
        
        self.model = None
        self.clone_prompt = None
        self.playback_thread = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        
        # Ensure voice directory exists
        os.makedirs(self.voice_dir, exist_ok=True)

    def run(self):
        # 1. Check/generate reference voice
        self._ensure_reference_voice()
        
        # 2. Initialize the Qwen3-TTS model
        try:
            self._load_model()
            
            if not self.model or not self.clone_prompt:
                print("[TTS System] Model or clone prompt missing. Worker will not start.")
                return

            # Start the playback thread
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
            # 3. Process the text queue (Generator Loop)
            print("[TTS System] Playback thread is active. Waiting for audio...", flush=True)
            print("[TTS System] Player queue is active. Waiting for sentences...", flush=True)
        finally:
            self.ready_event.set()
        
        while self.running:
            try:
                text = self.text_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            if text == "__STOP__":
                self.audio_queue.put(("__STOP__", 0))
                break
                
            self._generate_audio(text)
            self.text_queue.task_done()
            
        print("[TTS System] Generator queue stopped.")

    def _ensure_reference_voice(self):
        """Verifies reference voice exists, otherwise uses pyttsx3 to generate a default one."""
        wav_path = os.path.join(self.voice_dir, "voice.wav")
        txt_path = os.path.join(self.voice_dir, "voice.txt")
        
        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            print("[TTS System] Reference voice files missing. Creating default speaker reference...")
            try:
                import pyttsx3
                engine = pyttsx3.init()
                
                # Check for standard english voice
                voices = engine.getProperty('voices')
                for voice in voices:
                    if "EN-US" in voice.id.upper() or "ENGLISH" in voice.name.upper():
                        engine.setProperty('voice', voice.id)
                        break
                        
                default_text = "Hello, I am Kiri, your virtual desktop companion. Let's make something amazing together!"
                
                # Save to wav
                engine.save_to_file(default_text, wav_path)
                engine.runAndWait()
                
                # Save transcript
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(default_text)
                    
                print(f"[TTS System] Default voice.wav and voice.txt saved to '{self.voice_dir}/'")
            except Exception as e:
                print(f"[TTS System] Warning: Failed to generate default voice via pyttsx3: {e}")
                print("[TTS System] Please manually place voice.wav and voice.txt in config/voice/")

    def _load_model(self):
        """Loads the Qwen3-TTS model weights and pre-computes the speaker's embedding prompt."""
        print(f"[TTS System] Loading model '{self.model_id}' on {self.device} ({self.dtype})...")
        try:
            # Silence all warning outputs and downloads progress bars printed during load
            with silence_stdout_stderr():
                from qwen_tts import Qwen3TTSModel
                
                # Load model
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation="sdpa"
                )
            
            # Load reference paths
            wav_path = os.path.join(self.voice_dir, "voice.wav")
            txt_path = os.path.join(self.voice_dir, "voice.txt")
            prompt_path = os.path.join(self.voice_dir, "voice_prompt.pt")
            
            if os.path.exists(prompt_path):
                print(f"[TTS System] Loading pre-compiled speaker profile from '{prompt_path}'...")
                self.clone_prompt = torch.load(prompt_path, weights_only=False)
                print("[TTS System] Qwen3-TTS model loaded and pre-compiled clone prompt initialized successfully.")
            elif os.path.exists(wav_path) and os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
                
                print(f"[TTS System] Computing speaker clone features for reference text: '{ref_text[:50]}...'")
                with silence_stdout_stderr():
                    self.clone_prompt = self.model.create_voice_clone_prompt(
                        ref_audio=wav_path,
                        ref_text=ref_text
                    )
                # Auto-cache the prompt for next launch
                try:
                    torch.save(self.clone_prompt, prompt_path)
                    print(f"[TTS System] Compiled features cached to '{prompt_path}' for future runs.")
                except Exception as cache_err:
                    print(f"[TTS System] Warning: Failed to cache compiled features: {cache_err}")
                print("[TTS System] Qwen3-TTS model loaded and clone prompt initialized successfully.")
            else:
                print("[TTS System] Error: Cannot load clone prompt. Speaker profile or reference voice files are missing.")
                
        except Exception as e:
            print(f"[TTS System] Critical error loading Qwen3-TTS model: {e}")
            print("[TTS System] Speech playback will be disabled.")

    def _generate_audio(self, text: str):
        """Generates cloned speech waveform and puts it into the audio queue."""
        if not self.model or not self.clone_prompt:
            return
            
        try:
            # Generate waveform (wavs: np.ndarray or torch.Tensor, sr: int)
            wavs, sr = self.model.generate_voice_clone(
                text=[text],
                language=["English"],
                voice_clone_prompt=self.clone_prompt
            )
            
            # If wavs is returned as a list, extract the first element (batch size 1)
            if isinstance(wavs, list):
                raw_audio = wavs[0]
            else:
                raw_audio = wavs
                
            # Convert torch.Tensor to numpy array
            if hasattr(raw_audio, "cpu"):
                audio_data = raw_audio.cpu().numpy()
            else:
                audio_data = raw_audio
                
            # Squeeze dimensions if 2D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()
                
            self.audio_queue.put((audio_data, sr))
            
        except Exception as e:
            print(f"\n[TTS System] Error generating speech for '{text}': {e}")

    def _playback_loop(self):
        """Monitors the audio queue and plays speech waveforms sequentially."""
        while self.running:
            try:
                item = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            audio_data, sr = item
            if isinstance(audio_data, str) and audio_data == "__STOP__":
                self.audio_queue.task_done()
                break
                
            try:
                sd.play(audio_data, sr)
                sd.wait() # Wait until sentence playback completes
            except Exception as e:
                print(f"\n[TTS System] Playback error: {e}")
            finally:
                self.audio_queue.task_done()
                
        print("[TTS System] Playback loop stopped.")

    def enqueue_speech(self, text: str):
        """Appends a new sentence to the text queue for background generation."""
        if self.model and self.clone_prompt:
            self.text_queue.put(text)

    def get_status(self) -> str:
        """Returns the current status of the TTS system: 'preparing', 'speaking', or 'idle'."""
        if self.text_queue.unfinished_tasks > 0:
            return "preparing"
        elif self.audio_queue.unfinished_tasks > 0:
            return "speaking"
        else:
            return "idle"

    def wait_until_done(self):
        """Blocks until all text is generated and all audio has finished playing."""
        self.text_queue.join()
        self.audio_queue.join()

    def stop(self):
        """Signals the generator and player threads to terminate cleanly."""
        self.running = False
        self.text_queue.put("__STOP__")
