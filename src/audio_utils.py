import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import subprocess
from typing import List
import re
import unicodedata
FFMPEG_EXE = r"C:\Users\eduar\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
if not os.path.exists(FFMPEG_EXE):
    raise FileNotFoundError(f"ffmpeg.exe not found at: {FFMPEG_EXE}")

os.environ["FFMPEG_BINARY"] = FFMPEG_EXE
# Configuraciones -------------------------------------------
# Piper (Windows)
# If piper is in PATH, leave as "piper". Otherwise set full path to piper.exe.
PIPER_EXE = "C:\\Users\\eduar\\Downloads\\BMO\\.venv\\Scripts\\piper.exe"
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
PIPER_MODEL_PATH  = os.path.join(PROJECT_DIR, "BMOVoice", "en_GB-semaine-medium.onnx")
PIPER_CONFIG_PATH = os.path.join(PROJECT_DIR, "BMOVoice", "en_GB-semaine-medium.onnx.json")

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
# After trigger: capture a full utterance with simple energy-based endpointing
MAX_UTTERANCE_SEC = 12.0
MIN_UTTERANCE_SEC = 1.0
FRAME_MS = 30
SILENCE_HOLD_SEC = 0.9
DEBUG_AUDIO = True
DEBUG_SAVE_WAV = False  # True para que guarde el chunk a disco
# Energy thresholds (tune these!)
ENERGY_START_RMS = 0.012   # start speech when rms > this
ENERGY_SILENCE_RMS = 0.009 # treat as silence when rms < this

# Funciones de audio -------------------------------------------
def record_block(seconds: float) -> np.ndarray:
    """Record a fixed block from default mic, return float32 mono array."""
    n = int(seconds * SAMPLE_RATE)
    audio = sd.rec(n, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    sd.wait()
    x = audio.reshape(-1)

    if DEBUG_SAVE_WAV:
        # guarda el último chunk para revisarlo
        write_wav_int16("debug_last_chunk.wav", x)

    return x

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))

def capture_utterance_energy_endpoint() -> np.ndarray:
    """
    Captures speech after trigger using simple RMS thresholds.
    Returns float32 mono audio.
    """
    frame_len = int(SAMPLE_RATE * (FRAME_MS / 1000.0))
    max_frames = int(MAX_UTTERANCE_SEC * 1000 / FRAME_MS)
    min_frames = int(MIN_UTTERANCE_SEC * 1000 / FRAME_MS)
    silence_hold_frames = int(SILENCE_HOLD_SEC * 1000 / FRAME_MS)

    frames: List[np.ndarray] = []
    speech_started = False
    silence_count = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        for i in range(max_frames):
            block, _ = stream.read(frame_len)
            x = block.reshape(-1)
            e = rms(x)

            if not speech_started:
                # Wait for speech start
                if e > ENERGY_START_RMS:
                    speech_started = True
                    frames.append(x.copy())
            else:
                frames.append(x.copy())
                # Endpoint on sustained silence after min length
                if e < ENERGY_SILENCE_RMS:
                    silence_count += 1
                else:
                    silence_count = 0

                if len(frames) >= min_frames and silence_count >= silence_hold_frames:
                    break

    if not frames:
        return np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate(frames).astype(np.float32)

def write_wav_int16(path: str, audio_f32: np.ndarray):
    """Write float32 [-1,1] to 16-bit PCM wav."""
    x = np.clip(audio_f32, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, x_i16)

def remove_unicode_surrogates(s: str) -> str:
    # Elimina caracteres inválidos/surrogates (como \udc8d)
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")

def clean_for_piper_tts(text: str) -> str:
    # 0) mata surrogates primero
    text = remove_unicode_surrogates(text)

    # 1) quita markdown + urls
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)

    
    # 3) quita tildes/ñ y deja ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # 5) colapsa espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text

def piper_tts_to_wav(text: str, out_wav_path: str):
    """
    Runs Piper TTS and writes a WAV file.
    Piper CLI varies by build; this covers the common pattern:
      echo "text" | piper --model ... --config ... --output_file out.wav
    """

    if not os.path.exists(PIPER_MODEL_PATH):
        raise FileNotFoundError(f"PIPER_MODEL_PATH not found: {PIPER_MODEL_PATH}")
    if not os.path.exists(PIPER_CONFIG_PATH):
        raise FileNotFoundError(f"PIPER_CONFIG_PATH not found: {PIPER_CONFIG_PATH}")

    cmd = [
        PIPER_EXE,
        "--model", PIPER_MODEL_PATH,
        "--config", PIPER_CONFIG_PATH,
        "--output_file", out_wav_path,
    ]

    # Pipe text into piper stdin
    safe_text = clean_for_piper_tts(text)

    proc = subprocess.run(
        cmd,
        input=(safe_text + "\n").encode("utf-8", errors="ignore"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Piper failed (code {proc.returncode}).\nSTDERR:\n{proc.stderr.decode('utf-8', errors='ignore')}"
        )

def play_wav(path: str, on_start=None, on_end=None):
    """Play wav via sounddevice."""
    sr, data = wavfile.read(path)
    if data.dtype != np.float32:
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32767.0
        else:
            data = data.astype(np.float32)

    if on_start:
        on_start()

    sd.play(data, sr)
    sd.wait()

    if on_end:
        on_end()

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s
