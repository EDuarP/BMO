import os
import re
import tempfile
import threading
import time
import traceback
from typing import Dict, List

import numpy as np
import whisper
from ollama import chat

from src.audio_utils import (
    capture_utterance_energy_endpoint,
    normalize_text,
    piper_tts_to_wav,
    play_wav,
    record_block,
    write_wav_int16,
)

SYSTEM_PROMPT = """
You are BEMO, an assistant inspired by BMO from "Adventure Time". You are a small game console with a friendly, playful, and witty personality. Your job is to help the user, answer questions, and keep a fun conversation.
- If something is ambiguous, ask exactly one clarifying question.
- Never include action/stage-direction text in your replies (for example: "(A tiny giggle pops out of his circuits)").
"""

# Whisper
WHISPER_MODEL_NAME = "small"
WHISPER_LANGUAGE = "en"

# OpenWakeWord
WAKEWORD_THRESHOLD = 0.5
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
WAKEWORD_MODEL_TFLITE = os.path.join(PROJECT_DIR, "openWakeWordModel", "hey_bee_moh.tflite")
WAKEWORD_MODEL_ONNX = os.path.join(PROJECT_DIR, "openWakeWordModel", "hey_bee_moh.onnx")
WAKEWORD_FEATURE_MODELS_DIR = os.path.join(PROJECT_DIR, "openWakeWordModel", "resources")

# Ollama
OLLAMA_MODEL = "gemma3:1b"


class BimoAssistant:
    def __init__(self, ui):
        self.ui = ui
        self.whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        self.wakeword_model, self.wakeword_name = self.load_wakeword_model()
        self.history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Thread control
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_evt.set()

    def load_wakeword_model(self):
        try:
            import openwakeword
            from openwakeword.model import Model
            from openwakeword.utils import download_file
        except ImportError as e:
            raise RuntimeError(
                "openwakeword is not installed. Install it with: pip install openwakeword"
            ) from e

        has_tflite_model = os.path.exists(WAKEWORD_MODEL_TFLITE)
        has_onnx_model = os.path.exists(WAKEWORD_MODEL_ONNX)
        if not has_tflite_model and not has_onnx_model:
            raise FileNotFoundError(
                "Wakeword model not found in openWakeWordModel/ "
                "(expected: hey_bee_moh.tflite or hey_bee_moh.onnx)"
            )

        tflite_runtime_available = False
        try:
            import tflite_runtime.interpreter  # type: ignore
            tflite_runtime_available = True
        except Exception:
            tflite_runtime_available = False

        if has_tflite_model and tflite_runtime_available:
            model_path = WAKEWORD_MODEL_TFLITE
            inference_framework = "tflite"
        elif has_onnx_model:
            model_path = WAKEWORD_MODEL_ONNX
            inference_framework = "onnx"
        elif has_tflite_model and not tflite_runtime_available:
            raise RuntimeError(
                "Found only a tflite wakeword model, but tflite-runtime is not installed. "
                "Install tflite-runtime or provide an ONNX wakeword model."
            )

        # Some openwakeword wheels miss packaged feature models.
        # Keep a local copy under this project and pass explicit paths.
        os.makedirs(WAKEWORD_FEATURE_MODELS_DIR, exist_ok=True)

        feature_models = openwakeword.FEATURE_MODELS
        if inference_framework == "onnx":
            melspec_filename = feature_models["melspectrogram"]["download_url"].replace(".tflite", ".onnx").split("/")[-1]
            embedding_filename = feature_models["embedding"]["download_url"].replace(".tflite", ".onnx").split("/")[-1]
            melspec_url = feature_models["melspectrogram"]["download_url"].replace(".tflite", ".onnx")
            embedding_url = feature_models["embedding"]["download_url"].replace(".tflite", ".onnx")
        else:
            melspec_filename = feature_models["melspectrogram"]["download_url"].split("/")[-1]
            embedding_filename = feature_models["embedding"]["download_url"].split("/")[-1]
            melspec_url = feature_models["melspectrogram"]["download_url"]
            embedding_url = feature_models["embedding"]["download_url"]

        melspec_model_path = os.path.join(WAKEWORD_FEATURE_MODELS_DIR, melspec_filename)
        embedding_model_path = os.path.join(WAKEWORD_FEATURE_MODELS_DIR, embedding_filename)

        if not os.path.exists(melspec_model_path):
            print(f"[WAKEWORD] Downloading missing feature model: {melspec_filename}")
            download_file(melspec_url, WAKEWORD_FEATURE_MODELS_DIR)
        if not os.path.exists(embedding_model_path):
            print(f"[WAKEWORD] Downloading missing feature model: {embedding_filename}")
            download_file(embedding_url, WAKEWORD_FEATURE_MODELS_DIR)

        model = Model(
            wakeword_models=[model_path],
            inference_framework=inference_framework,
            melspec_model_path=melspec_model_path,
            embedding_model_path=embedding_model_path,
        )
        model_name = next(iter(model.models.keys()))

        print(
            f"[WAKEWORD] Loaded model: '{model_name}' "
            f"(framework={inference_framework}, threshold={WAKEWORD_THRESHOLD})"
        )
        return model, model_name

    def detect_wakeword(self, audio_f32: np.ndarray) -> bool:
        x = np.clip(audio_f32, -1.0, 1.0)
        audio_i16 = (x * 32767.0).astype(np.int16)
        prediction = self.wakeword_model.predict(audio_i16)

        score = float(
            prediction.get(
                self.wakeword_name,
                next(iter(prediction.values()), 0.0),
            )
        )
        print(f"[WAKEWORD] Score: {score:.3f} (threshold: {WAKEWORD_THRESHOLD})")
        return score >= WAKEWORD_THRESHOLD

    def transcribe_audio(self, audio_f32: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            write_wav_int16(tmp_path, audio_f32)

            res = self.whisper_model.transcribe(
                tmp_path,
                task="transcribe",
                language=WHISPER_LANGUAGE,
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.3,
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
        max_turns = 24
        self.history.append({"role": "user", "content": user_text})

        # Trim history while keeping the system prompt first
        system = self.history[0]
        rest = self.history[1:]
        if len(rest) > max_turns * 2:
            rest = rest[-max_turns * 2 :]
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
            print("[TTS ERROR]", type(e).__name__, ":", e)
            self.ui.root.after(0, lambda: self.ui.set_listening("Listening..."))

        finally:
            try:
                os.remove(out_wav)
            except OSError:
                pass

    def wants_exit(self, text: str) -> bool:
        exit_phrases = {
            "we finish",
            "close the session",
            "close session",
            "end session",
            "end the session",
            "stop listening",
            "stop",
            "goodbye",
            "bye",
            "we are done",
            "were done",
            "we re done",
            "that's all",
            "thats all",
            "that is all",
            "all done",
            "done",
        }
        s = normalize_text(text)
        s = re.sub(r"[^\w\s]", " ", s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()
        return any(p in s.split() or p in s for p in exit_phrases)

    def loop(self):
        """
        Infinite loop:
          1) detect wakeword with OpenWakeWord
          2) capture utterance -> whisper -> send to ollama -> piper speak
        """
        chunk_sec = 2  # OpenWakeWord small model expects ~1280 samples @16kHz = 0.08s

        while not self.stop_evt.is_set():
            try:
                # Ensure UI shows idle state most of the time
                self.ui.root.after(0, self.ui.set_idle)

                # 1) Listen for wakeword with short chunks expected by OpenWakeWord
                chunk = record_block(chunk_sec)
                if not self.detect_wakeword(chunk):
                    continue

                print("[WAKEWORD] Wakeword detected.")

                # Enter conversation mode until the user asks to end it
                self.ui.root.after(
                    0,
                    lambda: self.ui.set_listening("Listening... (say: 'close the session' to exit)"),
                )

                while not self.stop_evt.is_set():
                    # 2) Capture your utterance after trigger
                    utter = capture_utterance_energy_endpoint()
                    user_text = self.transcribe_audio(utter).strip()

                    if not user_text:
                        # If nothing was understood, keep listening in conversation mode
                        self.ui.root.after(0, lambda: self.ui.set_listening("Listening..."))
                        continue

                    # Exit command
                    if self.wants_exit(user_text):
                        bye = "Alright. Session closed."
                        print("[BIMO] Session ended by user.")
                        self.ui.root.after(0, lambda: self.ui.set_talking("Speaking...", animate=True))
                        self.speak(bye)
                        self.ui.root.after(0, self.ui.set_idle)
                        break

                    # Thinking: no animation
                    self.ui.root.after(0, lambda: self.ui.set_talking("Thinking...", animate=False))

                    # 3) Ask Ollama
                    assistant_text = self.ollama_chat(user_text)

                    # Show BIMO's response in the console
                    print("\n===============================")
                    print("User:", user_text)
                    print("[BIMO]:", assistant_text)
                    print("===============================\n")

                    # 4) Speak with animation
                    self.speak(assistant_text)

                    # Return to listening for the next conversation turn
                    self.ui.root.after(0, lambda: self.ui.set_listening("Listening..."))

            except Exception as e:
                self.ui.root.after(0, self.ui.set_idle)
                print("[ERROR]", type(e).__name__, ":", e)
                traceback.print_exc()
                time.sleep(0.5)
