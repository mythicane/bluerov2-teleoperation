"""
BlueROV2 PID / gain tuner.

Loads pid_configs.yaml, shows sliders + text inputs for each parameter,
and writes back to the file on Save.  Restart the controller after saving
for changes to take effect.

Run:  python utils/tune_pid.py
"""

import tkinter as tk
from pathlib import Path
import time
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "pid_configs.yaml"

# (section, key, display_label, lo, hi, resolution, unit)
PARAMS = [
    ("depth",   "kp",         "KP",          0.0,   3000.0, 1.0,   ""),
    ("depth",   "ki",         "KI",          0.0,    100.0, 0.1,   ""),
    ("depth",   "kd",         "KD",          0.0,    500.0, 1.0,   ""),
    ("depth",   "depth_rate", "Depth Rate",  0.0,      2.0, 0.01,  "m/s"),
    ("depth",   "depth_min",  "Depth Min",  -2.0,      0.0, 0.1,   "m"),
    ("depth",   "depth_max",  "Depth Max",   5.0,    200.0, 1.0,   "m"),
    ("lateral", "gain",       "Gain",        0.0,      1.0, 0.005, ""),
    ("lateral", "deadband",   "Deadband",    0.0,  10000.0, 0.005, ""),
]

SECTION_LABELS = {
    "depth":   "Depth Controller",
    "lateral": "Lateral / Yaw",
}


def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def save_config(rows: list) -> None:
    cfg = load_config()
    for row in rows:
        cfg.setdefault(row.section, {})[row.key] = round(row.get(), 6)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ── per-parameter row ─────────────────────────────────────────────────────────

class ParamRow:
    def __init__(self, parent, section, key, label, lo, hi, res, unit, value, row_idx,
                 on_change=None):
        self.section    = section
        self.key        = key
        self._lo        = lo
        self._hi        = hi
        self._on_change = on_change

        self._var = tk.DoubleVar(value=value)

        # label
        tk.Label(parent, text=label, anchor="w", width=13,
                 font=("Consolas", 10)).grid(row=row_idx, column=0, sticky="w", padx=(8, 4), pady=3)

        # slider
        self._scale = tk.Scale(
            parent,
            from_=lo, to=hi, resolution=res,
            orient="horizontal", variable=self._var,
            command=self._on_scale,
            length=340, showvalue=False,
            bg="#2b2b2b", fg="#dddddd",
            troughcolor="#444444", activebackground="#5599ff",
            highlightthickness=0, bd=0,
        )
        self._scale.grid(row=row_idx, column=1, padx=4, pady=3, sticky="ew")

        # text entry
        self._entry_var = tk.StringVar(value=self._fmt(value))
        self._entry = tk.Entry(
            parent, textvariable=self._entry_var,
            width=9, font=("Consolas", 10),
            bg="#1e1e1e", fg="#dddddd",
            insertbackground="#dddddd",
            relief="flat", bd=2,
        )
        self._entry.grid(row=row_idx, column=2, padx=(4, 2), pady=3)
        self._entry.bind("<Return>",   self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

        # unit
        tk.Label(parent, text=unit, anchor="w", width=5,
                 font=("Consolas", 10), fg="#888888").grid(
            row=row_idx, column=3, sticky="w", padx=(0, 8))

    @staticmethod
    def _fmt(v: float) -> str:
        return f"{v:.4f}".rstrip("0").rstrip(".")

    def _on_scale(self, _val):
        self._entry_var.set(self._fmt(self._var.get()))
        if self._on_change:
            self._on_change()

    def _on_entry(self, _event):
        try:
            v = float(self._entry_var.get())
            v = max(self._lo, min(self._hi, v))
            self._var.set(v)
            self._entry_var.set(self._fmt(v))
            if self._on_change:
                self._on_change()
        except ValueError:
            self._entry_var.set(self._fmt(self._var.get()))

    def get(self) -> float:
        return self._var.get()


# ── main window ───────────────────────────────────────────────────────────────

class TunerApp:
    _AUTOSAVE_DELAY_MS = 300   # debounce: write after this many ms of no changes

    def __init__(self, root: tk.Tk):
        self._root = root
        self._pending_save = None   # holds root.after() handle

        root.title("BlueROV2 PID Tuner  (live)")
        root.configure(bg="#2b2b2b")
        root.resizable(False, False)

        cfg = load_config()
        self._rows: list[ParamRow] = []

        # group params by section
        sections: dict[str, list] = {}
        for p in PARAMS:
            sections.setdefault(p[0], []).append(p)

        for section, params in sections.items():
            frame = tk.LabelFrame(
                root,
                text=f"  {SECTION_LABELS.get(section, section)}  ",
                font=("Consolas", 10, "bold"),
                bg="#2b2b2b", fg="#5599ff",
                labelanchor="nw", bd=1, relief="groove",
            )
            frame.pack(fill="x", padx=12, pady=(10, 0))
            frame.columnconfigure(1, weight=1)

            for i, (sec, key, label, lo, hi, res, unit) in enumerate(params):
                value = cfg.get(sec, {}).get(key, lo)
                row = ParamRow(frame, sec, key, label, lo, hi, res, unit, value, i,
                               on_change=self._schedule_autosave)
                self._rows.append(row)

        # button row
        btn_frame = tk.Frame(root, bg="#2b2b2b")
        btn_frame.pack(fill="x", padx=12, pady=10)

        btn_style = dict(font=("Consolas", 10), relief="flat", bd=0,
                         padx=14, pady=5, cursor="hand2")

        tk.Button(btn_frame, text="Save now", bg="#2266cc", fg="white",
                  activebackground="#3377dd", command=self._save,
                  **btn_style).pack(side="left", padx=(0, 6))

        tk.Button(btn_frame, text="Reload", bg="#444444", fg="#dddddd",
                  activebackground="#555555", command=self._reload,
                  **btn_style).pack(side="left")

        self._status = tk.Label(btn_frame, text=f"Loaded  {CONFIG_PATH.name}",
                                font=("Consolas", 9), fg="#888888", bg="#2b2b2b")
        self._status.pack(side="right", padx=4)

    def _schedule_autosave(self):
        if self._pending_save is not None:
            self._root.after_cancel(self._pending_save)
        self._pending_save = self._root.after(self._AUTOSAVE_DELAY_MS, self._autosave)
        self._status.config(text="Unsaved changes...", fg="#ccaa44")

    def _autosave(self):
        self._pending_save = None
        save_config(self._rows)
        self._status.config(text=f"Live  {time.strftime('%H:%M:%S')}", fg="#66cc66")

    def _save(self):
        if self._pending_save is not None:
            self._root.after_cancel(self._pending_save)
            self._pending_save = None
        save_config(self._rows)
        self._status.config(text=f"Saved  {time.strftime('%H:%M:%S')}", fg="#66cc66")

    def _reload(self):
        cfg = load_config()
        for row in self._rows:
            v = cfg.get(row.section, {}).get(row.key)
            if v is not None:
                row._var.set(v)
                row._entry_var.set(ParamRow._fmt(v))
        self._status.config(text=f"Reloaded  {time.strftime('%H:%M:%S')}", fg="#ccaa44")


def main():
    root = tk.Tk()
    TunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
