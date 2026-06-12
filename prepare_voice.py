import os
import sys
import warnings
import torch

# Suppress Hugging Face/PyTorch warnings and progress bars to keep terminal clean
from huggingface_hub.utils import disable_progress_bars
from transformers import logging as transformers_logging
disable_progress_bars()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
transformers_logging.set_verbosity_error()

from src.tts import silence_stdout_stderr

def compile_voice() -> bool:
    voice_dir = "config/voice"
    wav_path = os.path.join(voice_dir, "voice.wav")
    txt_path = os.path.join(voice_dir, "voice.txt")
    output_path = os.path.join(voice_dir, "voice_prompt.pt")
    
    if not os.path.exists(wav_path) or not os.path.exists(txt_path):
        print(f"Error: Missing speaker files. Please ensure both '{wav_path}' and '{txt_path}' exist.")
        return False
        
    with open(txt_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
        
    print(f"\n1. Reference transcript: '{ref_text}'")
    print("2. Loading Qwen3-TTS model on CUDA (torch.bfloat16)...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    
    try:
        with silence_stdout_stderr():
            from qwen_tts import Qwen3TTSModel
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=device,
                dtype=dtype,
                attn_implementation="sdpa"
            )
            
        print("3. Computing speaker embeddings and audio codes...")
        with silence_stdout_stderr():
            clone_prompt = model.create_voice_clone_prompt(
                ref_audio=wav_path,
                ref_text=ref_text
            )
            
        print(f"4. Saving speaker profile to '{output_path}'...")
        torch.save(clone_prompt, output_path)
        print("\n[Success] Speaker profile successfully compiled and saved!")
        print("Kiri will now load this profile instantly on launch with zero preparation overhead.")
        return True
    except Exception as e:
        print(f"\n[Error] Failed to compile speaker profile: {e}")
        return False

def main():
    print("==================================================")
    # Ensure UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("         Qwen3-TTS Speaker Profile Compiler       ")
    print("==================================================")
    
    success = compile_voice()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
