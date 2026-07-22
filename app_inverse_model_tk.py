"""Tkinter desktop interface for saved RF-trap inverse models."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from rf_trap_forward.export_prediction_dataset import load_prediction_model
from rf_trap_forward.predict_inverse import (
    DEFAULT_MODEL_PATH,
    INPUT_COLUMNS,
    InversePredictionBatch,
    format_prediction_text,
    load_minima_csv,
    parse_minima_string,
    predict_inverse,
    write_prediction_csv,
)


class InverseModelApp(ttk.Frame):
    """Small model-only desktop prediction form with CSV batch support."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=14)
        self.master = master
        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL_PATH))
        self.units = tk.StringVar(value="m")
        self.coordinate_values = [tk.StringVar() for _ in INPUT_COLUMNS]
        self.current_prediction: InversePredictionBatch | None = None
        self.status = tk.StringVar(value="Ready. No FEM solve or training is performed.")
        self._build()

    def _build(self) -> None:
        self.master.title("RF trap inverse-model prediction")
        self.master.minsize(820, 650)
        self.grid(sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(7, weight=1)

        ttk.Label(self, text="Inverse RF trap model", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        ttk.Label(self, text="Model file:").grid(row=1, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.model_path).grid(
            row=1, column=1, sticky="ew", padx=8
        )
        ttk.Button(self, text="Choose model...", command=self.choose_model).grid(
            row=1, column=2, sticky="e"
        )

        minima = ttk.LabelFrame(self, text="Three equilibrium/minimum positions", padding=10)
        minima.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        for column, label in enumerate(("Minimum", "x", "y")):
            ttk.Label(minima, text=label).grid(row=0, column=column, padx=4, sticky="w")
        for index in range(3):
            ttk.Label(minima, text=f"min{index + 1}").grid(
                row=index + 1, column=0, padx=4, pady=3, sticky="w"
            )
            ttk.Entry(minima, textvariable=self.coordinate_values[2 * index], width=20).grid(
                row=index + 1, column=1, padx=4, pady=3
            )
            ttk.Entry(
                minima, textvariable=self.coordinate_values[2 * index + 1], width=20
            ).grid(row=index + 1, column=2, padx=4, pady=3)
        ttk.Label(minima, text="Input units:").grid(row=1, column=3, padx=(18, 4))
        ttk.Combobox(
            minima,
            textvariable=self.units,
            values=("m", "mm"),
            state="readonly",
            width=8,
        ).grid(row=1, column=4, padx=4)

        controls = ttk.Frame(self)
        controls.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Button(controls, text="Predict", command=self.predict_entries).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(controls, text="Load and predict CSV...", command=self.load_csv).grid(
            row=0, column=1, padx=8
        )
        ttk.Button(controls, text="Save prediction CSV...", command=self.save_csv).grid(
            row=0, column=2, padx=8
        )

        explanation = (
            "Wolfram order: W1 upper-right, W2 lower-right, W3 upper-left, "
            "W4 lower-left. FEM transform: F1,F2,F3,F4 = -[W3,W1,W4,W2]."
        )
        ttk.Label(self, text=explanation, wraplength=770).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        ttk.Label(self, textvariable=self.status, wraplength=770).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        ttk.Label(self, text="Prediction output:").grid(
            row=6, column=0, columnspan=3, sticky="w"
        )
        self.output = tk.Text(self, height=22, wrap="none", font=("Consolas", 10))
        self.output.grid(row=7, column=0, columnspan=3, sticky="nsew")

    def choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose inverse model",
            filetypes=(("Joblib model", "*.joblib"), ("All files", "*.*")),
        )
        if path:
            self.model_path.set(path)
            self.status.set(f"Selected model: {path}")

    def predict_entries(self) -> None:
        try:
            value = ";".join(
                f"{self.coordinate_values[index].get()},{self.coordinate_values[index + 1].get()}"
                for index in range(0, 6, 2)
            )
            minima_m = parse_minima_string(value, self.units.get())
            self._run_prediction(minima_m)
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error))

    def load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Load minima CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            minima_m = load_minima_csv(path, self.units.get())
            self._show_first_input(minima_m[0])
            self._run_prediction(minima_m)
            self.status.set(
                f"Loaded and predicted {minima_m.shape[0]} row(s) from {path}. "
                "The output view shows at most five rows."
            )
        except Exception as error:
            messagebox.showerror("CSV prediction failed", str(error))

    def _show_first_input(self, minima_row_m: np.ndarray) -> None:
        scale = 1.0 if self.units.get() == "m" else 1.0e3
        for variable, value in zip(
            self.coordinate_values, scale * minima_row_m, strict=True
        ):
            variable.set(f"{value:.12g}")

    def _run_prediction(self, minima_m: object) -> None:
        model = load_prediction_model(Path(self.model_path.get()))
        self.current_prediction = predict_inverse(model, minima_m)
        text = format_prediction_text(self.current_prediction)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        warning_count = sum(bool(item) for item in self.current_prediction.warnings)
        if warning_count:
            self.status.set(
                f"Prediction complete: {warning_count} row(s) exceed the +/-500 um "
                "training range."
            )
        else:
            self.status.set("Prediction complete; all displacement coordinates are within +/-500 um.")

    def save_csv(self) -> None:
        if self.current_prediction is None:
            messagebox.showinfo("Nothing to save", "Run a prediction or load a CSV first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save prediction CSV",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            write_prediction_csv(self.current_prediction, path)
            self.status.set(f"Saved {self.current_prediction.minima_m.shape[0]} row(s) to {path}")
        except Exception as error:
            messagebox.showerror("Save failed", str(error))


def main() -> None:
    root = tk.Tk()
    InverseModelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
