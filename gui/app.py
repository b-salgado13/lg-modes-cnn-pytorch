"""Tkinter reproduction of the App Designer GUI (Appendix 6).

Reproduces the core workflow of `GUILaguerreGaussBeams`:

1. Load a trained checkpoint (any of the three network types).
2. Select a pattern image through a file picker.
3. Run the prediction and display the reconstructed mode + confidence,
   mirroring `PredictLGModeButtonPushed`.

The decorative fiber-propagation animation and reference mode thumbnails
from the original GUI depend on institution-specific static assets and are
intentionally left out; see the README for details.

Run with:
    python -m gui.app
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from PIL import Image, ImageTk

from src.mode_labels import parse_folder_name
from src.predict import _predict_image_model, _predict_mlp, load_checkpoint


class LGModeClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Laguerre-Gauss Mode Classifier (PyTorch)")
        self.geometry("520x640")
        self.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.checkpoint = None
        self.tk_image = None  # keep a reference alive for Tkinter

        self.checkpoint_path_var = tk.StringVar(value="No checkpoint selected")
        self.image_path_var = tk.StringVar(value="No image selected")

        self._build_widgets()

    def _build_widgets(self) -> None:
        pad = {"padx": 12, "pady": 6}

        tk.Label(self, text="Laguerre-Gauss Mode Classifier", font=("Helvetica", 16, "bold")).pack(**pad)

        tk.Label(self, text="1. Load a trained checkpoint (.pt)", font=("Helvetica", 12, "bold")).pack(
            **pad, anchor="w"
        )
        tk.Button(self, text="Choose checkpoint...", command=self.choose_checkpoint).pack(**pad, anchor="w")
        tk.Label(self, textvariable=self.checkpoint_path_var, wraplength=480, fg="gray").pack(
            **pad, anchor="w"
        )

        tk.Label(self, text="2. Select a pattern image", font=("Helvetica", 12, "bold")).pack(
            **pad, anchor="w"
        )
        tk.Button(self, text="Select pattern...", command=self.choose_image).pack(**pad, anchor="w")
        tk.Label(self, textvariable=self.image_path_var, wraplength=480, fg="gray").pack(**pad, anchor="w")

        self.image_label = tk.Label(self)
        self.image_label.pack(**pad)

        tk.Button(
            self, text="Predict LG mode", command=self.run_prediction, bg="#c6f2c6", font=("Helvetica", 11, "bold")
        ).pack(**pad)

        self.result_label = tk.Label(self, text="", font=("Helvetica", 14, "bold"), fg="#1a4d8f")
        self.result_label.pack(**pad)
        self.confidence_label = tk.Label(self, text="", font=("Helvetica", 11))
        self.confidence_label.pack(**pad)

    def choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt")])
        if not path:
            return
        try:
            self.model, self.checkpoint = load_checkpoint(path, self.device)
        except Exception as exc:  # noqa: BLE001 - surface any error to the user
            messagebox.showerror("Error loading checkpoint", str(exc))
            return
        self.checkpoint_path_var.set(path)

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JPEG images", "*.jpg *.jpeg *.JPG *.JPEG")]
        )
        if not path:
            return
        self.image_path_var.set(path)
        with Image.open(path) as img:
            preview = img.convert("RGB")
            preview.thumbnail((320, 320))
            self.tk_image = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.tk_image)

    def run_prediction(self) -> None:
        if self.model is None:
            messagebox.showwarning("No checkpoint", "Please load a checkpoint first.")
            return
        if self.image_path_var.get() == "No image selected":
            messagebox.showwarning("No image", "Please select a pattern image first.")
            return

        model_type = self.checkpoint["model_type"]
        image_path = self.image_path_var.get()
        try:
            if model_type == "simple_mlp":
                pred_idx, probs = _predict_mlp(self.model, self.checkpoint, image_path, self.device)
            elif model_type == "simple_cnn":
                pred_idx, probs = _predict_image_model(
                    self.model,
                    self.checkpoint,
                    image_path,
                    self.device,
                    channels=1,
                    image_size=self.checkpoint["input_size"],
                )
            else:  # transfer_learning
                pred_idx, probs = _predict_image_model(
                    self.model,
                    self.checkpoint,
                    image_path,
                    self.device,
                    channels=3,
                    image_size=self.checkpoint["image_size"],
                )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Prediction error", str(exc))
            return

        idx_to_class = {idx: name for name, idx in self.checkpoint["class_to_idx"].items()}
        class_name = idx_to_class[pred_idx]
        mode = parse_folder_name(class_name)

        self.result_label.configure(text=mode.describe())
        self.confidence_label.configure(text=f"Probabilidad = {probs[pred_idx].item() * 100:.2f}%")


def main() -> None:
    app = LGModeClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
