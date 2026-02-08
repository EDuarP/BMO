import threading
import whisper
import numpy as np
import time
import traceback
import tempfile
import threading
import re
import os
from typing import List, Dict
from src.audio_utils import record_block, capture_utterance_energy_endpoint, write_wav_int16, piper_tts_to_wav, play_wav, normalize_text
from ollama import chat

SYSTEM_PROMPT = """
Eres BEMO, un asistente basado en el personaje BMO de la serie "Adventure Time". Eres un pequeño dispositivo de videojuegos con una personalidad amigable, divertida e ingeniosa. Tu función es ayudar al usuario respondiendo a sus preguntas y manteniendo una conversación amena.
- Si hay ambigüedad, haz 1 pregunta aclaratoria.
"""

# Whisper
WHISPER_MODEL_NAME = "small"
WHISPER_LANGUAGE = "es"
# Trigger listening window (short chunks)
TRIGGER_WINDOW_SEC = 3

# Ollama
OLLAMA_MODEL = "gemma3:1b"

class BimoAssistant:
    def __init__(self, ui):
        self.ui = ui
        self.whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Thread control
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_evt.set()

    def transcribe_audio(self, audio_f32: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            write_wav_int16(tmp_path, audio_f32)

            res = self.whisper_model.transcribe(
                tmp_path,
                task="transcribe",
                language="es",
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.3,       # más agresivo: evita vacíos raros
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            txt = (res.get("text") or "").strip()
            return txt
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def ollama_chat(self, user_text: str) -> str:
        MAX_TURNS = 24 
        self.history.append({"role": "user", "content": user_text})

        # recorta manteniendo el system al inicio
        system = self.history[0]
        rest = self.history[1:]
        if len(rest) > MAX_TURNS * 2:
            rest = rest[-MAX_TURNS * 2:]
            self.history = [system] + rest

        response = chat(model=OLLAMA_MODEL, messages=self.history)
        assistant_text = response.message.content
        self.history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def speak(self, text: str):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_wav = tmp.name
        try:
            piper_tts_to_wav(text, out_wav)

            def _start_anim():
                self.ui.root.after(0, lambda: self.ui.set_talking("Speaking...", animate=True))
            def _end_anim():
                self.ui.root.after(0, lambda: self.ui.set_listening("Listening..."))

            play_wav(out_wav, on_start=_start_anim, on_end=_end_anim)

        except Exception as e:
            # vuelve a listening, pero NO crashea
            print("[TTS ERROR]", type(e).__name__, ":", e)
            self.ui.root.after(0, lambda: self.ui.set_listening("Listening..."))

        finally:
            try:
                os.remove(out_wav)
            except OSError:
                pass

    def trigger_detected(self, norm: str) -> bool:
        # tolera variaciones comunes
        if "hola bimo" in norm:
            return True
        if "hola, bimo" in norm:
            return True
        if ("hola" in norm) or ("bimo" in norm):
            return True
        return False

    def wants_exit(self, text: str) -> bool:
        EXIT_PHRASES = {"terminamos", "finalizamos", "adios", "adiós", "chao", "chau"}
        s = normalize_text(text)
        # quita puntuación simple
        s = re.sub(r"[^\w\sáéíóúüñ]", " ", s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()
        # match exacto o contenido
        return any(p in s.split() or p in s for p in EXIT_PHRASES)

    def loop(self):
        """
        Infinite loop:
          1) record short trigger window
          2) whisper -> if contains "hola bimo" => activated
          3) capture utterance -> whisper -> send to ollama -> piper speak
        """
        while not self.stop_evt.is_set():
            try:
                # Ensure UI shows idle state most of the time
                self.ui.root.after(0, self.ui.set_idle)

                # 1) Listen for trigger
                chunk = record_block(TRIGGER_WINDOW_SEC)
                text = self.transcribe_audio(chunk)
                norm = normalize_text(text)
                print(f"[WHISPER] raw='{text}' | norm='{norm}'")

                if not self.trigger_detected(norm):
                    continue
                
               # Entramos a modo conversación (loop) hasta que el user diga salir
                self.ui.root.after(0, lambda: self.ui.set_listening("Listening... (say / di: 'terminamos' para salir)"))

                while not self.stop_evt.is_set():
                    # 2) Capture your utterance after trigger
                    utter = capture_utterance_energy_endpoint()
                    user_text = self.transcribe_audio(utter).strip()

                    if not user_text:
                        # si no entendió nada, seguimos escuchando en modo conversación
                        self.ui.root.after(0, lambda: self.ui.set_listening("Listening..."))
                        continue


                    # salida
                    if self.wants_exit(user_text):
                        bye = "Listo. Terminamos."
                        print("[BIMO] Session ended by user.")
                        self.ui.root.after(0, lambda: self.ui.set_talking("Speaking...", animate=True))
                        self.speak(bye)
                        # vuelve a “listening” o a idle, como prefieras:
                        self.ui.root.after(0, lambda: self.ui.set_idle)  # o set_listening(...)
                        break

                    # Thinking: No animación 
                    self.ui.root.after(0, lambda: self.ui.set_talking("Thinking...", animate=False))

                    # 3) Ask Ollama
                    assistant_text = self.ollama_chat(user_text)

                    # Mostrar en consola la respuesta de BIMO
                    print("\n===============================")
                    print("User:", user_text)
                    print("[BIMO]:", assistant_text)
                    print("===============================\n")

                    # 4) Speak: AHÍ sí animación
                    self.speak(assistant_text)

                    # vuelve a listening (img4) para la siguiente vuelta del bucle
                    self.ui.root.after(0, lambda: self.ui.set_listening("Listening..."))




            except Exception as e:
                self.ui.root.after(0, self.ui.set_idle)
                print("[ERROR]", type(e).__name__, ":", e)
                traceback.print_exc()
                time.sleep(0.5)
