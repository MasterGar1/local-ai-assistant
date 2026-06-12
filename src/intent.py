import os
import re
from typing import List, Tuple

def detect_vision_trigger(prompt: str) -> bool:
    """Detects if the user prompt implies looking at the screen."""
    # Explicit commands
    if prompt.lower().startswith(("/see", "/screen")):
        return True
        
    # Semantic triggers (phrases indicating looking at the screen)
    keywords = [
        "what am i working on",
        "what am i looking at",
        "what is on my screen",
        "describe my screen",
        "see my screen",
        "look at my screen",
        "tell me what you see here",
        "what's on my screen",
        "think about what i am looking at",
        "can you see what i am looking at",
        "take a screenshot",
        "look at this"
    ]
    
    prompt_lower = prompt.lower()
    for kw in keywords:
        if kw in prompt_lower:
            return True
            
    return False

def find_paths_in_prompt(prompt: str) -> Tuple[List[str], List[str]]:
    """Scans the user prompt for existing file or directory paths.
    
    Returns:
        A tuple of (files, directories) absolute paths found in the prompt.
    """
    files = []
    dirs = []
    
    # 1. Look for quoted strings first (to capture spaces in paths)
    quoted_matches = re.findall(r'["\']([^"\']+)["\']', prompt)
    candidates = list(quoted_matches)
    
    # 2. Extract potential unquoted paths using regular expressions
    # This covers absolute Windows paths (e.g. C:\foo\bar or E:/foo/bar)
    win_paths = re.findall(r'([a-zA-Z]:[\\/][a-zA-Z0-9_\-\.\s\\/]+)', prompt)
    for p in win_paths:
        candidates.append(p.strip())
        
    # This covers typical unix paths or relative paths with file extensions
    # (e.g. /etc/hosts or ./src/gui.py or src/gui.py)
    unix_or_rel_paths = re.findall(r'([\.\~]?[\\/][a-zA-Z0-9_\-\.\/]+)', prompt)
    for p in unix_or_rel_paths:
        candidates.append(p.strip())
        
    # 3. Add individual tokens/words as potential paths
    words = prompt.split()
    for word in words:
        clean_word = word.strip('.,?!;()"\'-')
        if clean_word:
            candidates.append(clean_word)
            
    # Try merging contiguous words to catch unquoted paths with spaces (e.g., E:\My Project)
    for i in range(len(words)):
        for j in range(i + 1, min(i + 5, len(words) + 1)):
            phrase = " ".join(words[i:j]).strip('.,?!;()"\'-')
            if phrase:
                candidates.append(phrase)

    # 4. Filter candidates: check if they actually exist on disk
    seen = set()
    for path in candidates:
        # Avoid checking extremely short strings that are just words
        if len(path) < 3 and not (path.startswith("/") or path.startswith(".")):
            continue
            
        # Clean trailing slashes or backslashes to check existence cleanly
        clean_path = path.rstrip('\\/')
        if clean_path in seen:
            continue
        seen.add(clean_path)
        
        if os.path.exists(clean_path):
            abs_path = os.path.abspath(clean_path)
            if os.path.isfile(abs_path):
                if abs_path not in files:
                    files.append(abs_path)
            elif os.path.isdir(abs_path):
                if abs_path not in dirs:
                    dirs.append(abs_path)
                    
    return files, dirs
