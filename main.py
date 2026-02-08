"""
BIMO Voice Loop (Windows default mic)
- Always listens on the default Windows microphone.
- Uses OpenAI Whisper (model: "small") to detect the trigger phrase: "hola bimo".
- Only after the trigger is detected, it records your next utterance, transcribes it,
  sends it to Ollama (gemma3:1b), and speaks the response using Piper TTS (voice: semaine).
- UI:
    * Idle (disabled): shows IMAGE_1
    * While responding (LLM + TTS): alternates IMAGE_2 and IMAGE_3 every 0.5s

Requirements (pip):
  pip install openai-whisper sounddevice numpy scipy pillow ollama

Also required:
  - FFmpeg installed and in PATH (Whisper needs it).
  - Piper TTS installed and "piper" available in PATH (or set PIPER_EXE).
  - The "semaine" voice model (ONNX + JSON) downloaded locally.

Notes:
  - Whisper small is heavy for continuous streaming; this uses short chunk polling.
  - Tune TRIGGER_WINDOW_SEC and ENERGY_* thresholds for your environment.
"""
import tkinter as tk
from src.BimoUI import BimoUI
from src.BimoAssistant import BimoAssistant

def main():
    root = tk.Tk()
    ui = BimoUI(root)

    bimo = BimoAssistant(ui)
    bimo.start()

    try:
        root.mainloop()
    finally:
        bimo.stop()

if __name__ == "__main__":
    main()
