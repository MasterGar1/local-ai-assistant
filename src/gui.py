import os
import sys
import re
import time
import json
import math
from typing import List, Dict

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QSize
from PyQt6.QtGui import QPainter, QPixmap, QPainterPath, QColor, QIcon, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextBrowser, QGraphicsDropShadowEffect,
    QMenu, QSystemTrayIcon, QSizePolicy
)

import ollama
from src.brain import AIBrain
from src.memory import LongTermMemory
from src.tts import TTSManager

class RoundedLabel(QLabel):
    """A custom QLabel that renders its QPixmap with rounded corners."""
    def __init__(self, radius: int = 25, parent=None):
        super().__init__(parent)
        self.radius = radius
        self.original_pixmap = None

    def setPixmap(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        super().setPixmap(self.original_pixmap)

    def paintEvent(self, event):
        if self.original_pixmap and not self.original_pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Create a rounded rectangle path
            path = QPainterPath()
            rect = self.rect()
            path.addRoundedRect(
                float(rect.x()), float(rect.y()),
                float(rect.width()), float(rect.height()),
                float(self.radius), float(self.radius)
            )
            painter.setClipPath(path)
            
            # Draw scaled pixmap
            scaled_pix = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled_pix.width()) // 2
            y = (self.height() - scaled_pix.height()) // 2
            painter.drawPixmap(x, y, scaled_pix)
        else:
            super().paintEvent(event)

class ChatWorker(QThread):
    """Background worker thread to handle Ollama API requests and folder ingestion."""
    token_emitted = pyqtSignal(str)
    finished = pyqtSignal(str, list)  # full_response, memories
    ingest_finished = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, brain: AIBrain, memory: LongTermMemory, user_input: str, conversation_history: List[Dict[str, str]], mode: str = "chat", injected_file_context: str = ""):
        super().__init__()
        self.brain = brain
        self.memory = memory
        self.user_input = user_input
        self.conversation_history = conversation_history
        self.mode = mode  # "chat", "vision", "ingest"
        self.injected_file_context = injected_file_context
        self.screenshot_path = "memory/screenshot.png"

    def run(self):
        try:
            if self.mode == "chat":
                # RAG Search (long-term + temporary folder context)
                memories = self.memory.retrieve_context(self.user_input, limit=5)
                temp_memories = self.memory.retrieve_temp_context(self.user_input, limit=5)
                
                temp_history = self.conversation_history.copy()
                context_msgs = []
                
                if memories:
                    context_str = "\n".join([f"- {m['summary']}" for m in memories])
                    context_msgs.append(f"Relevant context from past conversations with the user:\n{context_str}")
                    
                if temp_memories:
                    temp_str = "\n".join([f"- [From {m['metadata']['source']}]: {m['summary']}" for m in temp_memories])
                    context_msgs.append(f"Relevant context from active folder/files:\n{temp_str}")
                    
                if self.injected_file_context:
                    context_msgs.append(self.injected_file_context)
                    
                if context_msgs:
                    context_msg_content = "\n\n".join(context_msgs) + (
                        "\n\nUse this context only if relevant to answer the user's input. "
                        "Do not mention that you retrieved this from memory or loaded files unless asked."
                    )
                    temp_history.append({"role": "system", "content": context_msg_content})
                
                temp_history.append({"role": "user", "content": self.user_input})
                
                # Stream response
                response_chunks = []
                for chunk in self.brain.generate_response(temp_history, stream=True):
                    self.token_emitted.emit(chunk)
                    response_chunks.append(chunk)
                    
                full_response = "".join(response_chunks)
                self.finished.emit(full_response, memories + temp_memories)
                
            elif self.mode == "vision":
                self.brain.ensure_vision_model_available()
                
                # Run the screen analysis privately (without streaming)
                vision_prompt = (
                    "Describe what is currently visible on this computer screen in detail: open windows, active applications, code, text, graphics, or activities."
                )
                vision_description = self.brain.generate_vision_response(vision_prompt, self.screenshot_path, stream=False)
                injected_vision_context = f"\n[Referenced Active Screen Contents]\nDescription of what is visible on the user's screen:\n{vision_description}\n"
                
                # Perform the same RAG search and prompt generation as chat mode
                memories = self.memory.retrieve_context(self.user_input, limit=5)
                temp_memories = self.memory.retrieve_temp_context(self.user_input, limit=5)
                
                temp_history = self.conversation_history.copy()
                context_msgs = []
                
                if memories:
                    context_str = "\n".join([f"- {m['summary']}" for m in memories])
                    context_msgs.append(f"Relevant context from past conversations with the user:\n{context_str}")
                    
                if temp_memories:
                    temp_str = "\n".join([f"- [From {m['metadata']['source']}]: {m['summary']}" for m in temp_memories])
                    context_msgs.append(f"Relevant context from active folder/files:\n{temp_str}")
                    
                # Inject visual context
                context_msgs.append(injected_vision_context)
                
                if context_msgs:
                    context_msg_content = "\n\n".join(context_msgs) + (
                        "\n\nUse this context only if relevant to answer the user's input. "
                        "Do not mention that you retrieved this from memory or loaded files unless asked."
                    )
                    temp_history.append({"role": "system", "content": context_msg_content})
                
                temp_history.append({"role": "user", "content": self.user_input})
                
                # Stream response using Kiri's main model (text generation)
                response_chunks = []
                for chunk in self.brain.generate_response(temp_history, stream=True):
                    self.token_emitted.emit(chunk)
                    response_chunks.append(chunk)
                    
                full_response = "".join(response_chunks)
                self.finished.emit(full_response, memories + temp_memories)
                
            elif self.mode == "ingest":
                dir_path = self.user_input
                indexed_count = self.memory.set_temp_directory(dir_path)
                self.ingest_finished.emit(indexed_count)
                
        except Exception as e:
            self.error_occurred.emit(str(e))

class KiriMainWindow(QMainWindow):
    """The translucent, glassmorphic, always-on-top PyQt6 desktop window for Kiri."""
    def __init__(self, brain: AIBrain, memory: LongTermMemory, tts: TTSManager):
        super().__init__()
        self.brain = brain
        self.memory = memory
        self.tts = tts
        self.conversation_history = []
        self.drag_position = QPoint()
        self.response_started = False
        self.waiting_for_tts = False
        
        # Configure frameless window, translucent background, and always-on-top behavior
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        self.init_ui()
        self.init_tray()
        
        # Animation & Status tracking QTimer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_avatar_effects)
        self.anim_timer.start(50)  # Tick every 50ms
        self.anim_counter = 0

    def init_ui(self):
        # Central Widget & Main Layout
        central_widget = QWidget(self)
        central_widget.setObjectName("central_widget")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15) # Leave space for the drop shadow glow!
        
        # 1. Glassmorphic Chat Panel Setup (Always visible)
        self.chat_panel = QWidget(self)
        self.chat_panel.setObjectName("chat_panel")
        self.chat_panel.setFixedSize(320, 220)
        
        chat_layout = QVBoxLayout(self.chat_panel)
        chat_layout.setContentsMargins(12, 12, 12, 12)
        chat_layout.setSpacing(8)
        
        # Title bar layout
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(4, 0, 4, 0)
        
        self.title_label = QLabel("KIRI ASSISTANT", self.chat_panel)
        self.title_label.setStyleSheet("""
            color: #a78bfa; 
            font-family: 'Segoe UI', 'Outfit', sans-serif;
            font-weight: 800; 
            font-size: 10px; 
            letter-spacing: 2px;
            background: transparent;
        """)
        
        self.status_dot = QLabel(self.chat_panel)
        self.status_dot.setFixedSize(8, 8)
        self.status_dot.setStyleSheet("""
            background-color: #10b981; 
            border-radius: 4px;
        """)
        
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.status_dot)
        
        chat_layout.addLayout(title_layout)
        
        self.speech_bubble = QTextBrowser(self.chat_panel)
        self.speech_bubble.setObjectName("speech_bubble")
        self.speech_bubble.setHtml(
            "<span style='color: #a7a7a7; font-style: italic;'>Kiri is ready. Type a message below...</span>"
        )
        chat_layout.addWidget(self.speech_bubble)
        
        self.chat_input = QLineEdit(self.chat_panel)
        self.chat_input.setObjectName("chat_input")
        self.chat_input.setPlaceholderText("Type a message...")
        self.chat_input.returnPressed.connect(self.submit_prompt)
        chat_layout.addWidget(self.chat_input)
        
        main_layout.addWidget(self.chat_panel)
        
        # Drop shadow glowing effect for breathing/state animation around the chat panel
        self.shadow_effect = QGraphicsDropShadowEffect(self)
        self.shadow_effect.setOffset(0, 0)
        self.shadow_effect.setBlurRadius(20)
        self.shadow_effect.setColor(QColor(139, 92, 246, 120))  # Violet glow
        self.chat_panel.setGraphicsEffect(self.shadow_effect)
        
        # Application-wide glassmorphism stylesheet
        self.setStyleSheet("""
            QWidget#central_widget {
                background: transparent;
            }
            QWidget#chat_panel {
                background-color: rgba(20, 20, 30, 220);
                border: 1.5px solid rgba(167, 139, 250, 80);
                border-radius: 16px;
            }
            QTextBrowser#speech_bubble {
                background-color: rgba(10, 10, 15, 120);
                border: 1px solid rgba(255, 255, 255, 15);
                border-radius: 10px;
                color: #e5e7eb;
                font-family: 'Segoe UI', 'Outfit', sans-serif;
                font-size: 13px;
                line-height: 1.4;
                padding: 6px;
            }
            QLineEdit#chat_input {
                background-color: rgba(10, 10, 15, 180);
                border: 1px solid rgba(167, 139, 250, 100);
                border-radius: 10px;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
                padding: 6px 10px;
            }
            QLineEdit#chat_input:focus {
                border: 1px solid rgba(244, 63, 94, 220);
            }
        """)
        
        self.setFixedSize(350, 250)

    def init_tray(self):
        # Configure tray icon menu for quick tasks
        self.tray_icon = QSystemTrayIcon(self)
        avatar_path = "config/kiri_avatar.png"
        if os.path.exists(avatar_path):
            self.tray_icon.setIcon(QIcon(avatar_path))
        else:
            # Create a simple icon if missing
            self.tray_icon.setIcon(QIcon.fromTheme("system-run"))
            
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show Kiri")
        show_action.triggered.connect(self.show_widget)
        
        hide_action = tray_menu.addAction("Hide Kiri")
        hide_action.triggered.connect(self.hide)
        
        tray_menu.addSeparator()
        
        clear_action = tray_menu.addAction("Clear Memory")
        clear_action.triggered.connect(self.clear_assistant_memory)
        
        recompile_action = tray_menu.addAction("Recompile Voice")
        recompile_action.triggered.connect(self.recompile_assistant_voice)
        
        tray_menu.addSeparator()
        
        exit_action = tray_menu.addAction("Exit")
        exit_action.triggered.connect(self.exit_app)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
        # Double click on tray toggles show/hide
        self.tray_icon.activated.connect(self.on_tray_activated)

    # --- Mouse Event Overrides for Draggability ---
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(event.position().toPoint())
            # Do not drag if the user clicked inside the text browser or input box
            if child in [self.chat_input, self.speech_bubble] or (self.speech_bubble and child == self.speech_bubble.viewport()):
                return
            
            # Store the click position relative to the top-left of the window
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.is_dragging = True
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if getattr(self, "is_dragging", False) and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            event.accept()

    def contextMenuEvent(self, event):
        # Right-click context menu directly on Kiri
        context_menu = QMenu(self)
        context_menu.setStyleSheet("""
            QMenu {
                background-color: rgb(30, 30, 40);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: rgb(139, 92, 246);
            }
        """)
        
        clear_action = context_menu.addAction("Clear Memory")
        clear_action.triggered.connect(self.clear_assistant_memory)
        
        recompile_action = context_menu.addAction("Recompile Voice")
        recompile_action.triggered.connect(self.recompile_assistant_voice)
        
        context_menu.addSeparator()
        
        exit_action = context_menu.addAction("Exit")
        exit_action.triggered.connect(self.exit_app)
        
        context_menu.exec(event.globalPos())

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_widget()

    def show_widget(self):
        self.show()
        self.raise_()
        self.activateWindow()

    # --- Animation & Visual Effects ---
    def update_avatar_effects(self):
        self.anim_counter += 1
        status = self.tts.get_status()
        
        # Check if we were waiting for TTS to finish speaking and we are now idle
        if getattr(self, "waiting_for_tts", False) and status == "idle":
            self.waiting_for_tts = False
            self.chat_input.setEnabled(True)
            self.chat_input.clear()
            self.chat_input.setFocus()
            
        # Gentle Breathing Scale / Animation
        breathing = math.sin(self.anim_counter * 0.08)
        
        if status == "preparing":
            # Rapid Cyan-Amber Pulsing
            pulse = math.sin(self.anim_counter * 0.25)
            blur = int(22 + 8 * pulse)
            color = QColor(34, 211, 238, 200)  # Cyan
            self.shadow_effect.setColor(color)
            self.shadow_effect.setBlurRadius(blur)
            
            # Status dot: Amber/Cyan pulsing
            self.status_dot.setStyleSheet(f"""
                background-color: rgb({int(130 + 120 * pulse)}, 211, 238); 
                border-radius: 4px;
            """)
        elif status == "speaking":
            # Dynamic Pink Pulsing
            pulse = math.sin(self.anim_counter * 0.18)
            blur = int(24 + 10 * pulse)
            color = QColor(244, 63, 94, 220)  # Neon Pink
            self.shadow_effect.setColor(color)
            self.shadow_effect.setBlurRadius(blur)
            
            # Status dot: Pink
            self.status_dot.setStyleSheet("""
                background-color: #f43f5e; 
                border-radius: 4px;
            """)
        else:
            # Gentle Violet breathing
            blur = int(14 + 4 * breathing)
            color = QColor(139, 92, 246, 120)  # Violet
            self.shadow_effect.setColor(color)
            self.shadow_effect.setBlurRadius(blur)
            
            # Status dot: Green breathing
            green_val = int(160 + 50 * breathing)
            self.status_dot.setStyleSheet(f"""
                background-color: rgb(16, {green_val}, 129); 
                border-radius: 4px;
            """)

    # --- RAG and Chat Ingestion logic ---
    def submit_prompt(self):
        prompt = self.chat_input.text().strip()
        if not prompt:
            return
            
        self.current_prompt = prompt
        self.response_started = False
        self.chat_input.setEnabled(False)
        self.speech_bubble.clear()
        
        # Intercept explicit commands
        if prompt.lower() in ["/exit", "/quit"]:
            self.exit_app()
            return
            
        elif prompt.lower() == "/help":
            help_html = (
                "<span style='color: #a78bfa; font-weight: bold; font-size: 11px;'>[Available Commands]</span><br>"
                "<span style='color: #22d3ee;'>/help</span> - Show this help menu<br>"
                "<span style='color: #22d3ee;'>/see</span> or <span style='color: #22d3ee;'>/screen</span> [query] - Analyze screenshot<br>"
                "<span style='color: #22d3ee;'>/file &lt;path&gt;</span> - Inject file into context<br>"
                "<span style='color: #22d3ee;'>/dir &lt;path&gt;</span> - Load folder temporarily<br>"
                "<span style='color: #22d3ee;'>/clear_dir</span> - Clear loaded folder<br>"
                "<span style='color: #22d3ee;'>/exit</span> or <span style='color: #22d3ee;'>/quit</span> - Save logs and exit"
            )
            self.speech_bubble.setHtml(help_html)
            self.chat_input.setEnabled(True)
            self.chat_input.clear()
            self.chat_input.setFocus()
            return
            
        elif prompt.lower().startswith(("/see", "/screen")):
            query_match = re.match(r'^/(?:see|screen)\s*(.*)$', prompt, re.IGNORECASE)
            query = query_match.group(1).strip() if query_match else ""
            if not query:
                query = "Describe this screenshot."
                
            self.speech_bubble.setHtml(
                "<span style='color: #22d3ee; font-weight: bold;'>[Vision Mode] Capturing screen...</span>"
            )
            
            try:
                from src.vision import capture_screen
                capture_screen("memory/screenshot.png")
                self.speech_bubble.setHtml(
                    "<span style='color: #22d3ee; font-weight: bold;'>[Vision Mode] Analyzing screen...</span>"
                )
                
                self.worker = ChatWorker(
                    self.brain, self.memory, query, self.conversation_history, mode="vision"
                )
                self.worker.token_emitted.connect(self.append_response_token)
                self.worker.finished.connect(self.on_chat_finished)
                self.worker.error_occurred.connect(self.on_worker_error)
                self.worker.start()
            except Exception as e:
                self.speech_bubble.setHtml(f"<span style='color: #ef4444;'>Failed screen capture: {e}</span>")
                self.chat_input.setEnabled(True)
                
        elif prompt.lower().startswith("/file"):
            path_match = re.match(r'^/file\s*(.*)$', prompt, re.IGNORECASE)
            fpath = path_match.group(1).strip() if path_match else ""
            fpath = fpath.strip('"\'')
            if not fpath or not os.path.exists(fpath) or not os.path.isfile(fpath):
                msg = f"File not found or invalid path: {fpath}"
                self.speech_bubble.setHtml(f"<span style='color: #ef4444;'>{msg}</span>")
                self.chat_input.setEnabled(True)
            else:
                from src.ingest import extract_text_from_file
                text = extract_text_from_file(fpath)
                filename = os.path.basename(fpath)
                injected_file_context = f"\n[Referenced File Context - {filename}]\nPath: {fpath}\nContent:\n```\n{text[:25000]}\n```\n"
                self.speech_bubble.setHtml(
                    f"<span style='color: #22d3ee;'>Injected file '{filename}' into context. Thinking...</span>"
                )
                friendly_prompt = f"Please analyze and tell me about the file '{filename}'."
                self.current_prompt = friendly_prompt
                self.worker = ChatWorker(
                    self.brain, self.memory, friendly_prompt, self.conversation_history, mode="chat", injected_file_context=injected_file_context
                )
                self.worker.token_emitted.connect(self.append_response_token)
                self.worker.finished.connect(self.on_chat_finished)
                self.worker.error_occurred.connect(self.on_worker_error)
                self.worker.start()
                
        elif prompt.lower().startswith("/dir"):
            path_match = re.match(r'^/dir\s*(.*)$', prompt, re.IGNORECASE)
            dpath = path_match.group(1).strip() if path_match else ""
            dpath = dpath.strip('"\'')
            if not dpath or not os.path.exists(dpath) or not os.path.isdir(dpath):
                msg = f"Directory not found or invalid path: {dpath}"
                self.speech_bubble.setHtml(f"<span style='color: #ef4444;'>{msg}</span>")
                self.chat_input.setEnabled(True)
            else:
                self.current_dir_path = dpath
                self.speech_bubble.setHtml(
                    f"<span style='color: #a78bfa; font-weight: bold;'>[Ingest System] Scanning and indexing directory '{os.path.basename(dpath)}'...</span>"
                )
                self.worker = ChatWorker(
                    self.brain, self.memory, dpath, [], mode="ingest"
                )
                self.worker.ingest_finished.connect(self.on_ingest_finished)
                self.worker.error_occurred.connect(self.on_worker_error)
                self.worker.start()
                
        elif prompt.lower().startswith(("/clear_dir", "/clear_context")):
            self.memory.clear_temp_context()
            msg = "Cleared temporary context directory."
            self.speech_bubble.setHtml(f"<span style='color: #10b981;'>{msg}</span>")
            self.tts.enqueue_speech(msg)
            self.chat_input.setEnabled(True)
            self.chat_input.clear()
            self.chat_input.setFocus()
            
        else:
            # Implicit conversational triggers
            from src.intent import detect_vision_trigger, find_paths_in_prompt
            
            # 1. Implicit Vision Trigger Check
            if detect_vision_trigger(prompt):
                self.speech_bubble.setHtml(
                    "<span style='color: #22d3ee; font-weight: bold;'>[Vision Mode] Capturing screen...</span>"
                )
                try:
                    from src.vision import capture_screen
                    capture_screen("memory/screenshot.png")
                    self.speech_bubble.setHtml(
                        "<span style='color: #22d3ee; font-weight: bold;'>[Vision Mode] Analyzing screen...</span>"
                    )
                    self.worker = ChatWorker(
                        self.brain, self.memory, prompt, self.conversation_history, mode="vision"
                    )
                    self.worker.token_emitted.connect(self.append_response_token)
                    self.worker.finished.connect(self.on_chat_finished)
                    self.worker.error_occurred.connect(self.on_worker_error)
                    self.worker.start()
                except Exception as e:
                    self.speech_bubble.setHtml(f"<span style='color: #ef4444;'>Failed screen capture: {e}</span>")
                    self.chat_input.setEnabled(True)
                return
                
            # 2. File/Folder Mention Scanner
            detected_files, detected_dirs = find_paths_in_prompt(prompt)
            injected_file_context = ""
            
            folder_keywords = ["folder", "directory", "context", "use", "set", "load", "reference", "look at"]
            has_folder_kw = any(kw in prompt.lower() for kw in folder_keywords)
            
            if detected_dirs and has_folder_kw:
                dpath = detected_dirs[0]
                self.current_dir_path = dpath
                self.speech_bubble.setHtml(
                    f"<span style='color: #a78bfa; font-weight: bold;'>[Ingest System] Automatically loading directory '{os.path.basename(dpath)}'...</span>"
                )
                self.worker = ChatWorker(
                    self.brain, self.memory, dpath, [], mode="ingest"
                )
                self.worker.ingest_finished.connect(self.on_ingest_finished)
                self.worker.error_occurred.connect(self.on_worker_error)
                self.worker.start()
                return
                
            if detected_files:
                from src.ingest import extract_text_from_file
                for fpath in detected_files[:3]:
                    text = extract_text_from_file(fpath)
                    if text.strip():
                        filename = os.path.basename(fpath)
                        print(f"[{self.brain.name} Context] Automatically loaded file '{filename}' into context.")
                        injected_file_context += f"\n[Referenced File Context - {filename}]\nPath: {fpath}\nContent:\n```\n{text[:25000]}\n```\n"

            # Standard chat loop with RAG (long-term + temp)
            self.worker = ChatWorker(
                self.brain, self.memory, prompt, self.conversation_history, mode="chat", injected_file_context=injected_file_context
            )
            self.worker.token_emitted.connect(self.append_response_token)
            self.worker.finished.connect(self.on_chat_finished)
            self.worker.error_occurred.connect(self.on_worker_error)
            self.worker.start()

    def append_response_token(self, token: str):
        if not getattr(self, "response_started", False):
            self.speech_bubble.clear()
            self.response_started = True
        current_text = self.speech_bubble.toPlainText()
        # Maintain HTML style and append token
        self.speech_bubble.setPlainText(current_text + token)
        
        # Auto-scroll to bottom
        sb = self.speech_bubble.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_chat_finished(self, full_response: str, memories: list):
        # Add to local GUI session history
        self.conversation_history.append({"role": "user", "content": self.current_prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})
        
        # Print logs to console
        print(f"\nUser: {self.current_prompt}")
        print(f"{self.brain.name}: {full_response}")
        if memories:
            print(f"[{self.brain.name} Memory] Loaded {len(memories)} context items.")
            
        # Clean up markdown formatting and speak
        has_speech = False
        if full_response:
            clean_text = re.sub(r'[*_`#\-+>]', '', full_response)
            clean_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_text)
            clean_text = clean_text.strip()
            if clean_text:
                self.tts.enqueue_speech(clean_text)
                has_speech = True
                
        if has_speech:
            self.waiting_for_tts = True
        else:
            self.chat_input.setEnabled(True)
            self.chat_input.clear()
            self.chat_input.setFocus()

    def on_ingest_finished(self, count: int):
        dpath = getattr(self, "current_dir_path", "the directory")
        msg = f"Successfully loaded directory '{os.path.basename(dpath)}' ({count} files indexed) into temporary context memory!"
        self.speech_bubble.setPlainText(msg)
        print(f"\n{self.brain.name}: {msg}")
        self.tts.enqueue_speech(msg)
        
        self.chat_input.setEnabled(True)
        self.chat_input.clear()
        self.chat_input.setFocus()

    def on_worker_error(self, err_msg: str):
        self.speech_bubble.setHtml(f"<span style='color: #ef4444;'>Error: {err_msg}</span>")
        print(f"[GUI Error] {err_msg}")
        self.chat_input.setEnabled(True)

    # --- System tray and utilities actions ---
    def clear_assistant_memory(self):
        print("\n[Menu] Clearing memory...")
        self.memory.clear_memory()
        logs_dir = "memory/logs"
        if os.path.exists(logs_dir):
            for filename in os.listdir(logs_dir):
                file_path = os.path.join(logs_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass
        self.speech_bubble.setHtml(
            "<span style='color: #10b981;'>Memory and session logs cleared successfully!</span>"
        )

    def recompile_assistant_voice(self):
        print("\n[Menu] Recompiling speaker profile...")
        self.speech_bubble.setHtml(
            "<span style='color: #f59e0b;'>Recompiling speaker profile... Please wait.</span>"
        )
        QApplication.processEvents()
        try:
            import prepare_voice
            success = prepare_voice.compile_voice()
            if success:
                self.speech_bubble.setHtml(
                    "<span style='color: #10b981;'>Speaker profile recompiled successfully! Restart app to load updated voice.</span>"
                )
            else:
                self.speech_bubble.setHtml(
                    "<span style='color: #ef4444;'>Voice compilation failed. Check console.</span>"
                )
        except Exception as e:
            self.speech_bubble.setHtml(
                "<span style='color: #ef4444;'>Voice compilation failed: " + str(e) + "</span>"
            )

    def exit_app(self):
        # Trigger clean exit memory extraction
        if self.conversation_history:
            print(f"\n[{self.brain.name} Memory] Extracting facts from GUI conversation session...")
            self.speech_bubble.setHtml(
                "<span style='color: #6366f1; font-weight: bold;'>Saving session memories...</span>"
            )
            QApplication.processEvents()
            
            # Format and run summary facts extraction
            conversation_text = ""
            for msg in self.conversation_history:
                role = "User" if msg["role"] == "user" else "Kiri"
                conversation_text += f"{role}: {msg['content']}\n"
                
            extraction_prompt = (
                "Analyze the following conversation between the User and the AI assistant (Kiri).\n"
                "Extract a list of discrete, key facts, preferences, user hobbies, details about the user, or tasks/reminders that the User shared.\n"
                "Rules:\n"
                "1. Each fact must be a short, standalone sentence.\n"
                "2. Keep them completely objective and factual.\n"
                "3. Format the output as a simple JSON list of strings: [\"fact 1\", \"fact 2\", ...].\n"
                "4. If no specific facts were shared, return an empty list: [].\n\n"
                f"Conversation:\n{conversation_text}\n\n"
                "Output JSON only:"
            )
            
            try:
                summary_response = ollama.chat(
                    model=self.brain.model_name,
                    messages=[{"role": "user", "content": extraction_prompt}],
                    stream=False
                )
                raw_output = summary_response["message"]["content"].strip()
                # Use main's parser helper if possible, or simple regex
                from main import parse_json_list
                facts = parse_json_list(raw_output)
                
                if facts:
                    print(f"[{self.brain.name} Memory] Extracted {len(facts)} facts from GUI session:")
                    history_json = json.dumps(self.conversation_history, ensure_ascii=False)
                    for fact in facts:
                        print(f"  - {fact}")
                        self.memory.add_memory(fact, history_json)
                
                # Save logs
                logs_dir = "memory/logs"
                os.makedirs(logs_dir, exist_ok=True)
                log_filename = f"{logs_dir}/chat_gui_{int(time.time())}.json"
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
                print(f"[{self.brain.name} Memory] Session logs saved to {log_filename}")
            except Exception as e:
                print(f"[{self.brain.name} Memory] Failed to save GUI session: {e}")
                
        # Shut down speech manager cleanly
        print(f"\n[{self.brain.name} Memory] Waiting for speech playback to finish...")
        self.tts.wait_until_done()
        self.tts.stop()
        self.tray_icon.hide()
        QApplication.quit()

def launch_gui(brain: AIBrain, memory: LongTermMemory, tts: TTSManager):
    """Launches the PyQt6 application loop."""
    # Ensure a QApplication exists
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    window = KiriMainWindow(brain, memory, tts)
    window.show_widget()
    sys.exit(app.exec())
