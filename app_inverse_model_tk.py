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


def copy_text_to_clipboard(
    clipboard_clear: object,
    clipboard_append: object,
    text: str,
) -> None:
    """Copy text through Tk-compatible clipboard callbacks.

    Keeping this tiny operation outside the window class makes the copy behavior
    testable without opening a desktop window.
    """

    clipboard_clear()
    clipboard_append(text)


class InverseModelApp(ttk.Frame):
    """Small model-only desktop prediction form with CSV batch support."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=14)
        self.master = master
        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL_PATH))
        self.units = tk.StringVar(value="m")
        self.auto_sort_minima = tk.BooleanVar(value=True)
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
        ttk.Checkbutton(
            minima,
            text="Auto-sort minima before prediction",
            variable=self.auto_sort_minima,
        ).grid(row=4, column=1, columnspan=4, padx=4, pady=(8, 0), sticky="w")

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
        ttk.Button(controls, text="Copy output", command=self.copy_output).grid(
            row=0, column=3, padx=8
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
        self.output = tk.Text(
            self, height=22, wrap="none", font=("Consolas", 10), takefocus=True
        )
        self.output.grid(row=7, column=0, columnspan=3, sticky="nsew")
        self.output.bind("<Control-Key-c>", self._copy_selected_output)
        self.output.bind("<Control-Key-C>", self._copy_selected_output)

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
            self._run_prediction(minima_m)
            self.status.set(
                f"Loaded and predicted {minima_m.shape[0]} row(s) from {path}. "
                f"Auto-sort minima: {'enabled' if self.auto_sort_minima.get() else 'disabled'}. "
                "The output view shows at most five rows."
            )
        except Exception as error:
            messagebox.showerror("CSV prediction failed", str(error))

    def _show_first_input(self, minima_row_m: np.ndarray) -> None:
        scale = 1.0 if self.units.get() == "m" else 1.0e3
        for variable, value in zip(
            self.coordinate_values, scale * np.asarray(minima_row_m).reshape(6), strict=True
        ):
            variable.set(f"{value:.12g}")

    def _run_prediction(self, minima_m: object) -> None:
        model = load_prediction_model(Path(self.model_path.get()))
        self.current_prediction = predict_inverse(
            model, minima_m, sort_minima=self.auto_sort_minima.get()
        )
        self._show_first_input(self.current_prediction.minima_m[0])
        text = format_prediction_text(self.current_prediction)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        warning_count = sum(bool(item) for item in self.current_prediction.warnings)
        sort_label = "enabled" if self.current_prediction.auto_sort_enabled else "disabled"
        if warning_count:
            self.status.set(
                f"Prediction complete; auto-sort minima: {sort_label}. "
                f"{warning_count} row(s) exceed the +/-500 um training range."
            )
        else:
            self.status.set(
                "Prediction complete; "
                f"auto-sort minima: {sort_label}; all displacement coordinates are "
                "within +/-500 um."
            )

    def _copy_selected_output(self, _event: object = None) -> str:
        """Copy the selected output text for the standard Ctrl+C shortcut."""

        try:
            selected = self.output.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return "break"
        copy_text_to_clipboard(
            self.master.clipboard_clear,
            self.master.clipboard_append,
            selected,
        )
        self.master.update()
        return "break"

    def copy_output(self) -> None:
        """Copy the complete prediction output to the clipboard."""

        text = self.output.get("1.0", tk.END).rstrip("\n")
        if not text:
            messagebox.showinfo("Nothing to copy", "Run a prediction first.")
            return
        copy_text_to_clipboard(
            self.master.clipboard_clear,
            self.master.clipboard_append,
            text,
        )
        self.master.update()
        self.status.set("Copied full prediction output to clipboard.")

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
