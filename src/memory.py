import os
import time
from typing import List, Dict, Any
import chromadb
import ollama

class LongTermMemory:
    """Manages ChromaDB persistent vector database for long-term memory retrieval."""
    
    def __init__(self, db_dir: str = "memory/chroma", embedding_model: str = "nomic-embed-text"):
        self.db_dir = db_dir
        self.embedding_model = embedding_model
        
        # Ensure database directory exists
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name="conversation_memory"
        )
        self.temp_collection = self.client.get_or_create_collection(
            name="temp_context_memory"
        )
        self.active_temp_dir = None
        
    def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding using Ollama's local embedding model."""
        try:
            # Support both older client 'embeddings' and newer client 'embed' methods
            if hasattr(ollama, 'embed'):
                response = ollama.embed(model=self.embedding_model, input=text)
                return response["embeddings"][0]
            else:
                response = ollama.embeddings(model=self.embedding_model, prompt=text)
                return response["embedding"]
        except Exception as e:
            # Fallback check inside error block: try alternative method in case of attribute resolution discrepancies
            try:
                if hasattr(ollama, 'embeddings'):
                    response = ollama.embeddings(model=self.embedding_model, prompt=text)
                    return response["embedding"]
                else:
                    response = ollama.embed(model=self.embedding_model, input=text)
                    return response["embeddings"][0]
            except Exception as inner_e:
                print(f"[Memory System] Error generating embedding with '{self.embedding_model}': {inner_e}")
                # Return standard 768-dim zero vector for nomic-embed-text fallback
                return [0.0] * 768

    def add_memory(self, summary: str, full_history_json: str):
        """Adds a conversation summary to long-term memory."""
        import uuid
        mem_id = f"mem_{uuid.uuid4().hex}"
        embedding = self._get_embedding(summary)
        
        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[{
                "timestamp": time.time(),
                "full_history": full_history_json
            }]
        )
        print(f"[Memory System] Saved memory: '{summary[:60]}...'")

    def clear_memory(self):
        """Clears all persistent conversation memories from ChromaDB."""
        try:
            self.client.delete_collection("conversation_memory")
            self.collection = self.client.get_or_create_collection(
                name="conversation_memory"
            )
            print("[Memory System] Successfully cleared all conversation memories from vector database.")
            return True
        except Exception as e:
            print(f"[Memory System] Error clearing memories: {e}")
            return False

    def retrieve_context(self, query: str, limit: int = 5, distance_threshold: float = 1.2) -> List[Dict[str, Any]]:
        """Queries the vector DB for relevant past conversations, filtering by relevance distance."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
            
        if count == 0:
            return []
            
        query_embedding = self._get_embedding(query)
        # Handle querying database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, count)
        )
        
        memories = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(documents)
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                if dist <= distance_threshold:
                    memories.append({
                        "summary": doc,
                        "metadata": meta,
                        "distance": dist
                    })
        return memories

    def set_temp_directory(self, dir_path: str) -> int:
        """Indexes text-based files in the given directory into temporary context memory.
        
        Returns the number of files successfully indexed.
        """
        import uuid
        from src.ingest import extract_text_from_file, chunk_text
        
        # Reset temp collection
        try:
            self.client.delete_collection("temp_context_memory")
        except Exception:
            pass
        self.temp_collection = self.client.get_or_create_collection(
            name="temp_context_memory"
        )
        self.active_temp_dir = os.path.abspath(dir_path)
        
        # Scan for files
        allowed_exts = {
            ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".ini", ".cfg",
            ".js", ".html", ".css", ".pdf", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h"
        }
        ignored_dirs = {
            ".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist",
            ".idea", ".vscode", "memory"
        }
        
        files_to_index = []
        for root, dirs, files in os.walk(dir_path):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            for file in files:
                _, ext = os.path.splitext(file.lower())
                if ext in allowed_exts:
                    full_path = os.path.join(root, file)
                    try:
                        # Cap file size to avoid processing massive logs/data (e.g. max 250KB)
                        if os.path.getsize(full_path) < 250000:
                            files_to_index.append(full_path)
                    except Exception:
                        pass
                        
        # Limit to first 150 files to prevent performance issues
        files_to_index = files_to_index[:150]
        
        indexed_count = 0
        for fpath in files_to_index:
            try:
                text = extract_text_from_file(fpath)
                if not text.strip():
                    continue
                    
                chunks = chunk_text(text)
                rel_path = os.path.relpath(fpath, dir_path)
                
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"temp_{uuid.uuid4().hex}"
                    meta = {
                        "source": rel_path,
                        "full_path": fpath,
                        "chunk_index": idx,
                        "timestamp": time.time()
                    }
                    self.temp_collection.add(
                        ids=[chunk_id],
                        embeddings=[self._get_embedding(chunk)],
                        documents=[chunk],
                        metadatas=[meta]
                    )
                indexed_count += 1
            except Exception as e:
                print(f"[Memory System] Failed to index temp file '{fpath}': {e}")
                
        return indexed_count

    def clear_temp_context(self):
        """Clears all temporary directory contexts and resets active_temp_dir."""
        try:
            self.client.delete_collection("temp_context_memory")
        except Exception:
            pass
        self.temp_collection = self.client.get_or_create_collection(
            name="temp_context_memory"
        )
        self.active_temp_dir = None
        print("[Memory System] Successfully cleared temporary folder context.")

    def retrieve_temp_context(self, query: str, limit: int = 5, distance_threshold: float = 1.2) -> List[Dict[str, Any]]:
        """Queries the temporary folder context collection for relevant text chunks."""
        try:
            count = self.temp_collection.count()
        except Exception:
            count = 0
            
        if count == 0:
            return []
            
        query_embedding = self._get_embedding(query)
        results = self.temp_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, count)
        )
        
        memories = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(documents)
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                if dist <= distance_threshold:
                    memories.append({
                        "summary": doc,
                        "metadata": meta,
                        "distance": dist
                    })
        return memories
