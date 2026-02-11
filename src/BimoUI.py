
import os

import tkinter as tk
from PIL import Image, ImageTk
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Quitar src del path para acceder a los recursos
BASE_DIR = os.path.dirname(BASE_DIR)


# UI animation
ANIM_PERIOD_SEC = 0.2
IMAGE_1_PATH = os.path.join(BASE_DIR, "BMOFace", "ojos_cerrados.png")
IMAGE_2_PATH = os.path.join(BASE_DIR, "BMOFace", "boca_abierta.png")
IMAGE_3_PATH = os.path.join(BASE_DIR, "BMOFace", "boca_cerrada.png")
IMAGE_4_PATH = os.path.join(BASE_DIR, "BMOFace", "rostro_feliz.png")
# =========================
# UI State
# =========================

@dataclass
class UIState:
    mode: str  # "idle" or "talking"
    last_text: str = ""


class BimoUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BIMO")
        self.root.geometry("520x520")

        self.img1 = ImageTk.PhotoImage(Image.open(IMAGE_1_PATH).resize((480, 480)))
        self.img2 = ImageTk.PhotoImage(Image.open(IMAGE_2_PATH).resize((480, 480)))
        self.img3 = ImageTk.PhotoImage(Image.open(IMAGE_3_PATH).resize((480, 480)))
        self.img4 = ImageTk.PhotoImage(Image.open(IMAGE_4_PATH).resize((480, 480)))


        self.label = tk.Label(root, image=self.img1)
        self.label.pack(pady=10)

        self.status = tk.Label(root, text="Idle (say: 'Hola BIMO')", font=("Segoe UI", 11))
        self.status.pack()

        self.state = UIState(mode="idle")
        self._anim_flip = False
        self._anim_running = False

        self._tick()

    def set_idle(self):
        self.state.mode = "idle"
        self.status.configure(text="Idle (say: 'Hey BMO')")
        self.label.configure(image=self.img1)
        self._anim_running = False
    
    def set_listening(self, msg: str = "Listening..."):
        self.state.mode = "listening"
        self.status.configure(text=msg)
        self._anim_running = False
        self.label.configure(image=self.img4)


    def set_talking(self, msg: str = "Talking...", animate: bool = True):
        self.state.mode = "talking"
        self.status.configure(text=msg)
        self._anim_running = bool(animate)
        if not self._anim_running:
            # si no animas, deja una imagen fija (por ejemplo img4 o img3)
            self.label.configure(image=self.img4)


    def _tick(self):
        # Animation loop: if talking, alternate img2/img3 every 0.5 sec
        if self._anim_running:
            self._anim_flip = not self._anim_flip
            self.label.configure(image=self.img2 if self._anim_flip else self.img3)
            self.root.after(int(ANIM_PERIOD_SEC * 1000), self._tick)
        else:
            self.root.after(200, self._tick)

