
# Transformer-Empowered Acoustic Monitoring Framework
# Integrating TAO Tree Classifier for Industry 4.0 Applications
# ─────────────────────────────────────────────────────────────


import os
import sys
import time
import threading
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import librosa
import librosa.display
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, precision_score, accuracy_score,
                             f1_score, recall_score, classification_report,
                             confusion_matrix)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import joblib

# Optional heavy imports – loaded lazily to keep startup fast
_torch = None
_transformers = None

# ─── Constants ───────────────────────────────────────────────
LABELS        = ['Air leak', 'Background noise', 'Idling', 'Normal', 'Oil leak']
MODEL_FOLDER  = "model"
HUBERT_NAME   = "facebook/hubert-base-ls960"

# ─── Palette ─────────────────────────────────────────────────
CLR = dict(
    bg        = "#0D1117",
    panel     = "#161B22",
    card      = "#1C2128",
    border    = "#30363D",
    accent    = "#58A6FF",      # electric blue
    accent2   = "#3FB950",      # green success
    warn      = "#F85149",      # red alert
    yellow    = "#D29922",
    text      = "#C9D1D9",
    text_dim  = "#8B949E",
    text_head = "#FFFFFF",
    gradient1 = "#58A6FF",
    gradient2 = "#388BFD",
)

STEP_COLORS = ["#58A6FF", "#3FB950", "#D29922", "#F78166", "#BC8CFF", "#39D353"]

os.makedirs(MODEL_FOLDER, exist_ok=True)

# ─── File paths ──────────────────────────────────────────────
X_file        = os.path.join(MODEL_FOLDER, "X.npy")
Y_file        = os.path.join(MODEL_FOLDER, "Y.npy")
X_hub_file    = os.path.join(MODEL_FOLDER, "X_hubert.npy")
Y_hub_file    = os.path.join(MODEL_FOLDER, "Y_hubert.npy")
Tao_hub_path  = os.path.join(MODEL_FOLDER, "Tao_on_HuBERT.pkl")
logreg_path   = os.path.join(MODEL_FOLDER, "LogisticRegression.pkl")
lda_path      = os.path.join(MODEL_FOLDER, "LDAClassifier.pkl")

# ─── Global state ────────────────────────────────────────────
app_state = dict(
    dataset_path  = None,
    categories    = [],
    X             = None,  Y       = None,
    X_train       = None,  X_test  = None,
    y_train       = None,  y_test  = None,
    X_hubert      = None,  Y_hubert= None,
    Tao_hubert    = None,
    step_done     = [False]*6,   # Track which steps completed
)


# ══════════════════════════════════════════════════════════════
#  Utility – thread-safe UI helpers
# ══════════════════════════════════════════════════════════════

def _get_device():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


class App(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("AcoustiQ Monitor — Industry 4.0")
        self.configure(bg=CLR["bg"])
        self.geometry("1280x820")
        self.minsize(1100, 700)

        self._init_fonts()
        self._build_layout()
        self._animate_status_dot()

    # ── Fonts ─────────────────────────────────────────────────
    def _init_fonts(self):
        from tkinter.font import Font
        self.fnt_h1  = ("Consolas", 18, "bold")
        self.fnt_h2  = ("Consolas", 13, "bold")
        self.fnt_btn = ("Consolas", 11, "bold")
        self.fnt_log = ("Courier New", 10)
        self.fnt_mono= ("Consolas", 10)
        self.fnt_big = ("Consolas", 28, "bold")
        self.fnt_lbl = ("Consolas", 10)

    # ── Master layout ─────────────────────────────────────────
    def _build_layout(self):
        # ── Top header ──────────────────────────────────────
        self._build_header()

        # ── Main body: sidebar + content ────────────────────
        body = tk.Frame(self, bg=CLR["bg"])
        body.pack(fill="both", expand=True)

        sidebar = self._build_sidebar(body)
        sidebar.pack(side="left", fill="y", padx=(8, 0), pady=8)

        right = tk.Frame(body, bg=CLR["bg"])
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # Workflow stepper
        self._build_stepper(right)

        # Cards row
        cards_frame = tk.Frame(right, bg=CLR["bg"])
        cards_frame.pack(fill="x", pady=(6, 0))
        self._build_metric_cards(cards_frame)

        # Log + waveform panes
        panes = tk.Frame(right, bg=CLR["bg"])
        panes.pack(fill="both", expand=True, pady=6)
        self._build_log_pane(panes)
        self._build_wave_pane(panes)

        # Status bar
        self._build_statusbar()

    # ── Header ────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg="#0D1B2A", height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Blinking status dot
        self._dot_var = tk.StringVar(value="●")
        self._dot_clr = [CLR["accent2"], CLR["accent"]]
        self._dot_idx = 0
        dot_lbl = tk.Label(hdr, textvariable=self._dot_var,
                           bg="#0D1B2A", font=("Consolas", 16, "bold"),
                           fg=CLR["accent2"])
        dot_lbl.pack(side="left", padx=(16, 6), pady=12)
        self._dot_label = dot_lbl

        tk.Label(
            hdr,
            text="AcoustiQ Monitor",
            bg="#0D1B2A", fg=CLR["text_head"],
            font=("Consolas", 17, "bold"),
        ).pack(side="left", pady=12)

        tk.Label(
            hdr,
            text="Industry 4.0 · HuBERT + TAO Tree Classifier",
            bg="#0D1B2A", fg=CLR["text_dim"],
            font=("Consolas", 10),
        ).pack(side="left", padx=(12, 0), pady=16)

        # Right info
        self._device_var = tk.StringVar(value="Device: CPU")
        tk.Label(hdr, textvariable=self._device_var,
                 bg="#0D1B2A", fg=CLR["yellow"],
                 font=self.fnt_lbl).pack(side="right", padx=16)
        threading.Thread(target=self._detect_device, daemon=True).start()

    def _detect_device(self):
        try:
            import torch
            dev = "CUDA · " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        except:
            dev = "CPU"
        self.after(0, lambda: self._device_var.set(f"Device: {dev}"))

    # ── Sidebar ───────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=CLR["panel"], width=230,
                      relief="flat", bd=0,
                      highlightbackground=CLR["border"],
                      highlightthickness=1)
        sb.pack_propagate(False)

        # Logo area
        logo_f = tk.Frame(sb, bg=CLR["panel"], pady=16)
        logo_f.pack(fill="x")
        tk.Label(logo_f, text="⚙", bg=CLR["panel"], fg=CLR["accent"],
                 font=("Consolas", 26, "bold")).pack()
        tk.Label(logo_f, text="W O R K F L O W", bg=CLR["panel"],
                 fg=CLR["text_dim"], font=("Consolas", 8, "bold")).pack(pady=(2, 0))

        tk.Frame(sb, bg=CLR["border"], height=1).pack(fill="x", padx=12, pady=6)

        # Step colours
        step_palette = [
            "#388BFD",  # blue   – Upload
            "#3FB950",  # green  – MFCC
            "#D29922",  # yellow – Split
            "#BC8CFF",  # purple – Logistic
            "#F78166",  # coral  – LDA
            "#39D353",  # lime   – HuBERT
        ]

        steps = [
            ("1", "Upload Dataset",   self.cmd_upload),
            ("2", "MFCC Extraction",  self.cmd_mfcc),
            ("3", "Split Data",       self.cmd_split),
            ("4", "Train Logistic",   self.cmd_train_logistic),
            ("5", "Train LDA",        self.cmd_train_lda),
            ("6", "Train HuBERT+TAO", self.cmd_train_hubert),
        ]

        self._step_btns = []

        def _make_step_btn(sb, i, num, label, cmd, accent):
            """Each step is a full-width styled button card."""
            # Outer wrapper gives the left coloured border effect
            wrapper = tk.Frame(sb, bg=accent, pady=0)
            wrapper.pack(fill="x", padx=12, pady=4)

            inner = tk.Frame(wrapper, bg=CLR["card"], cursor="hand2")
            inner.pack(fill="x", padx=(3, 0))   # 3px left accent strip

            # Badge circle canvas
            badge = tk.Canvas(inner, width=28, height=28,
                              bg=CLR["card"], highlightthickness=0)
            badge.pack(side="left", padx=(8, 6), pady=8)
            badge.create_oval(2, 2, 26, 26, fill=CLR["border"], outline="")
            badge.create_text(14, 14, text=num,
                              fill=CLR["text_dim"], font=("Consolas", 9, "bold"))

            # Label
            lbl = tk.Label(inner, text=label, bg=CLR["card"],
                           fg=CLR["text"], font=("Consolas", 10, "bold"),
                           anchor="w", cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True, pady=8)

            # Chevron arrow
            arrow = tk.Label(inner, text="›", bg=CLR["card"],
                             fg=CLR["text_dim"], font=("Consolas", 14),
                             cursor="hand2")
            arrow.pack(side="right", padx=8)

            # Bind click on all sub-widgets
            for widget in (inner, lbl, arrow, badge):
                widget.bind("<Button-1>", lambda e, c=cmd: c())
                widget.bind("<Enter>", lambda e, w=inner, a=accent: (
                    w.configure(bg=CLR["bg"]),
                    [c.configure(bg=CLR["bg"]) for c in w.winfo_children()
                     if isinstance(c, (tk.Label, tk.Canvas))]
                ))
                widget.bind("<Leave>", lambda e, w=inner, a=accent: (
                    w.configure(bg=CLR["card"]),
                    [c.configure(bg=CLR["card"]) for c in w.winfo_children()
                     if isinstance(c, (tk.Label, tk.Canvas))]
                ))

            self._step_btns.append((badge, lbl, i, accent, inner))

        for i, (num, label, cmd) in enumerate(steps):
            _make_step_btn(sb, i, num, label, cmd, step_palette[i])

        tk.Frame(sb, bg=CLR["border"], height=1).pack(fill="x", padx=12, pady=10)

        # ── Predict — prominent CTA ──────────────────────────
        pred_f = tk.Frame(sb, bg=CLR["accent"], pady=0)
        pred_f.pack(fill="x", padx=12, pady=4)
        pred_inner = tk.Frame(pred_f, bg="#1c4a8a", cursor="hand2")
        pred_inner.pack(fill="x")
        tk.Label(pred_inner, text="▶  Predict Audio",
                 bg="#1c4a8a", fg="white",
                 font=("Consolas", 11, "bold"),
                 cursor="hand2", pady=11).pack(fill="x")
        pred_inner.bind("<Button-1>", lambda e: self.cmd_predict())
        for w in pred_inner.winfo_children():
            w.bind("<Button-1>", lambda e: self.cmd_predict())
        pred_inner.bind("<Enter>", lambda e: pred_inner.configure(bg=CLR["accent"]))
        pred_inner.bind("<Leave>", lambda e: pred_inner.configure(bg="#1c4a8a"))

        # ── Utility buttons ──────────────────────────────────
        for txt, cmd in [("⊘  Clear Log", self.log_clear),
                         ("◫  Show Charts", self.cmd_show_charts)]:
            ub = tk.Button(sb, text=txt,
                           bg=CLR["card"], fg=CLR["text_dim"],
                           activebackground="#252b34",
                           activeforeground=CLR["text"],
                           font=("Consolas", 9, "bold"),
                           relief="flat", bd=0, cursor="hand2",
                           pady=7, command=cmd)
            ub.pack(fill="x", padx=12, pady=2)

        return sb

    # ── Stepper (top of content) ──────────────────────────────
    def _build_stepper(self, parent):
        f = tk.Frame(parent, bg=CLR["bg"])
        f.pack(fill="x", pady=(0, 4))
        self._stepper_labels = []
        names = ["Upload", "MFCC", "Split", "Logistic", "LDA", "HuBERT+TAO"]
        for i, name in enumerate(names):
            col_f = tk.Frame(f, bg=CLR["bg"])
            col_f.pack(side="left", expand=True)

            lbl_num = tk.Label(col_f, text=f"0{i+1}",
                               bg=CLR["bg"], fg=CLR["border"],
                               font=("Consolas", 16, "bold"))
            lbl_num.pack()
            lbl_name = tk.Label(col_f, text=name,
                                bg=CLR["bg"], fg=CLR["text_dim"],
                                font=("Consolas", 9))
            lbl_name.pack()
            self._stepper_labels.append((lbl_num, lbl_name))

            if i < len(names)-1:
                tk.Frame(f, bg=CLR["border"], height=2, width=18).pack(
                    side="left", pady=8)

    def _mark_step_done(self, step_idx):
        app_state["step_done"][step_idx] = True
        color = STEP_COLORS[step_idx]

        # Update top stepper
        num_lbl, name_lbl = self._stepper_labels[step_idx]
        num_lbl.configure(fg=color)
        name_lbl.configure(fg=color)

        # Update sidebar card badge -> show checkmark
        badge, lbl, idx, accent, inner = self._step_btns[step_idx]
        badge.delete("all")
        badge.create_oval(2, 2, 26, 26, fill=accent, outline="")
        badge.create_text(14, 14, text="✓", fill="#000000",
                          font=("Consolas", 9, "bold"))
        lbl.configure(fg=CLR["text_head"])

    # ── Metric cards ──────────────────────────────────────────
    def _build_metric_cards(self, parent):
        self._metric_vars = {}
        metrics = [
            ("Accuracy",  "acc",  CLR["accent"]),
            ("Precision", "prec", CLR["accent2"]),
            ("Recall",    "rec",  CLR["yellow"]),
            ("F1-Score",  "f1",   "#BC8CFF"),
        ]
        for title, key, color in metrics:
            card = tk.Frame(parent, bg=CLR["card"],
                            highlightbackground=CLR["border"],
                            highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=4)

            tk.Label(card, text=title, bg=CLR["card"],
                     fg=CLR["text_dim"], font=("Consolas", 9)).pack(pady=(8, 0))
            var = tk.StringVar(value="─ ─ ─")
            self._metric_vars[key] = var
            tk.Label(card, textvariable=var, bg=CLR["card"],
                     fg=color, font=("Consolas", 20, "bold")).pack(pady=(0, 8))

    def _update_metrics(self, acc, prec, rec, f1):
        self._metric_vars["acc"].set(f"{acc:.1f}%")
        self._metric_vars["prec"].set(f"{prec:.1f}%")
        self._metric_vars["rec"].set(f"{rec:.1f}%")
        self._metric_vars["f1"].set(f"{f1:.1f}%")

    # ── Log pane ──────────────────────────────────────────────
    def _build_log_pane(self, parent):
        lf = tk.Frame(parent, bg=CLR["panel"],
                      highlightbackground=CLR["border"],
                      highlightthickness=1)
        lf.pack(side="left", fill="both", expand=True, padx=(0, 4))

        hdr = tk.Frame(lf, bg=CLR["panel"])
        hdr.pack(fill="x", padx=8, pady=6)
        tk.Label(hdr, text="◉ CONSOLE OUTPUT", bg=CLR["panel"],
                 fg=CLR["accent"], font=("Consolas", 9, "bold")).pack(side="left")

        self.log_text = tk.Text(
            lf, bg="#090D12", fg=CLR["text"],
            font=self.fnt_log, relief="flat", bd=0,
            insertbackground=CLR["accent"],
            selectbackground=CLR["accent"],
        )
        self.log_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Configure tags for coloured output
        self.log_text.tag_config("ok",   foreground=CLR["accent2"])
        self.log_text.tag_config("err",  foreground=CLR["warn"])
        self.log_text.tag_config("info", foreground=CLR["accent"])
        self.log_text.tag_config("dim",  foreground=CLR["text_dim"])
        self.log_text.tag_config("head", foreground=CLR["yellow"],
                                  font=("Consolas", 10, "bold"))

        sb = tk.Scrollbar(lf, command=self.log_text.yview,
                          bg=CLR["panel"], troughcolor=CLR["bg"])
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self.log("AcoustiQ Monitor initialised.", tag="info")
        self.log(f"Model folder: {os.path.abspath(MODEL_FOLDER)}", tag="dim")

    # ── Waveform / mini chart pane ────────────────────────────
    def _build_wave_pane(self, parent):
        wf = tk.Frame(parent, bg=CLR["panel"],
                      highlightbackground=CLR["border"],
                      highlightthickness=1)
        wf.pack(side="left", fill="both", expand=True)

        hdr = tk.Frame(wf, bg=CLR["panel"])
        hdr.pack(fill="x", padx=8, pady=6)
        tk.Label(hdr, text="◉ WAVEFORM VIEWER", bg=CLR["panel"],
                 fg=CLR["accent"], font=("Consolas", 9, "bold")).pack(side="left")

        self._wave_fig = Figure(figsize=(4.5, 3.5), facecolor="#090D12")
        self._wave_ax  = self._wave_fig.add_subplot(111)
        self._wave_ax.set_facecolor("#090D12")
        self._wave_ax.tick_params(colors=CLR["text_dim"])
        for sp in self._wave_ax.spines.values():
            sp.set_color(CLR["border"])
        self._wave_ax.set_xlabel("Time (s)", color=CLR["text_dim"], fontsize=8)
        self._wave_ax.set_ylabel("Amplitude", color=CLR["text_dim"], fontsize=8)
        self._wave_ax.set_title("No file loaded", color=CLR["text_dim"], fontsize=9)

        self._wave_canvas = FigureCanvasTkAgg(self._wave_fig, master=wf)
        self._wave_canvas.get_tk_widget().pack(fill="both", expand=True,
                                               padx=6, pady=(0, 6))

    # ── Status bar ────────────────────────────────────────────
    def _build_statusbar(self):
        sb = tk.Frame(self, bg="#0D1B2A", height=26)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(sb, textvariable=self._status_var,
                 bg="#0D1B2A", fg=CLR["text_dim"],
                 font=("Consolas", 9)).pack(side="left", padx=12)

        self._progress = ttk.Progressbar(sb, mode="indeterminate",
                                          length=120, style="custom.Horizontal.TProgressbar")
        self._progress.pack(side="right", padx=12, pady=4)

        s = ttk.Style()
        s.theme_use("clam")
        s.configure("custom.Horizontal.TProgressbar",
                    troughcolor=CLR["border"], background=CLR["accent"],
                    thickness=6)

    def set_status(self, msg):
        self._status_var.set(msg)

    def start_busy(self):
        self._progress.start(12)
        self.set_status("Working…")

    def stop_busy(self):
        self._progress.stop()
        self.set_status("Ready")

    # ── Animations ────────────────────────────────────────────
    def _animate_status_dot(self):
        self._dot_idx ^= 1
        self._dot_label.configure(fg=self._dot_clr[self._dot_idx])
        self.after(800, self._animate_status_dot)

    # ── Log helpers ───────────────────────────────────────────
    def log(self, msg, tag=""):
        ts  = time.strftime("[%H:%M:%S] ")
        self.log_text.insert("end", ts, "dim")
        self.log_text.insert("end", msg + "\n", tag or "")
        self.log_text.see("end")

    def log_clear(self):
        self.log_text.delete("1.0", "end")

    def log_head(self, msg):
        self.log_text.insert("end", "\n" + "─"*56 + "\n", "dim")
        self.log_text.insert("end", f"  {msg}\n", "head")
        self.log_text.insert("end", "─"*56 + "\n", "dim")
        self.log_text.see("end")

    # ── Waveform updater ──────────────────────────────────────
    def _plot_waveform(self, audio, sr, title=""):
        ax = self._wave_ax
        ax.clear()
        ax.set_facecolor("#090D12")
        t = np.linspace(0, len(audio)/sr, len(audio))
        ax.plot(t, audio, color=CLR["accent"], linewidth=0.6, alpha=0.9)
        ax.fill_between(t, audio, alpha=0.15, color=CLR["accent"])
        ax.axhline(0, color=CLR["border"], linewidth=0.5)
        ax.set_title(title or "Waveform", color=CLR["text"], fontsize=9,
                     fontfamily="Consolas")
        ax.set_xlabel("Time (s)", color=CLR["text_dim"], fontsize=8)
        ax.set_ylabel("Amplitude", color=CLR["text_dim"], fontsize=8)
        ax.tick_params(colors=CLR["text_dim"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(CLR["border"])
        self._wave_fig.tight_layout(pad=0.4)
        self._wave_canvas.draw()

    # ══════════════════════════════════════════════════════════
    #  COMMAND handlers (run heavy work in daemon threads)
    # ══════════════════════════════════════════════════════════

    def _run_in_thread(self, fn):
        self.start_busy()
        def wrapper():
            try:
                fn()
            except Exception as e:
                self.after(0, lambda: self.log(f"ERROR: {e}", "err"))
            finally:
                self.after(0, self.stop_busy)
        threading.Thread(target=wrapper, daemon=True).start()

    # 1. Upload ───────────────────────────────────────────────
    def cmd_upload(self):
        folder = filedialog.askdirectory(title="Select dataset root folder")
        if not folder:
            return
        app_state["dataset_path"] = folder
        cats = [d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))]
        app_state["categories"] = cats

        self.log_head("DATASET UPLOAD")
        self.log(f"Path     : {folder}", "ok")
        self.log(f"Classes  : {len(cats)}", "info")
        for c in cats:
            n = len([f for f in os.listdir(os.path.join(folder, c))
                     if f.lower().endswith(".wav")])
            self.log(f"  ├─ {c:<25} {n} wav files", "dim")
        self._mark_step_done(0)

    # 2. MFCC ─────────────────────────────────────────────────
    def cmd_mfcc(self):
        self._run_in_thread(self._mfcc_task)

    def _mfcc_task(self):
        self.after(0, lambda: self.log_head("MFCC FEATURE EXTRACTION"))

        if os.path.exists(X_file) and os.path.exists(Y_file):
            app_state["X"] = np.load(X_file, allow_pickle=True)
            app_state["Y"] = np.load(Y_file, allow_pickle=True)
            self.after(0, lambda: self.log("Loaded cached X / Y arrays.", "ok"))
            self.after(0, lambda: self._mark_step_done(1))
            return

        path = app_state["dataset_path"]
        if not path:
            self.after(0, lambda: self.log("No dataset path. Run Upload first.", "err"))
            return

        cats = app_state["categories"]
        X_list, Y_list = [], []

        for root, dirs, files in os.walk(path):
            for fname in files:
                if not fname.lower().endswith(".wav") or "Thumbs.db" in fname:
                    continue
                fp = os.path.join(root, fname)
                try:
                    audio, sr = librosa.load(fp, sr=None)
                    mfccs     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    zc        = librosa.feature.zero_crossing_rate(y=audio)
                    sc        = librosa.feature.spectral_centroid(y=audio, sr=sr)
                    chroma    = librosa.feature.chroma_stft(y=audio, sr=sr)
                    bw        = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
                    contrast  = librosa.feature.spectral_contrast(y=audio, sr=sr)
                    rms       = librosa.feature.rms(y=audio)
                    rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
                    tonnetz   = librosa.feature.tonnetz(y=audio, sr=sr)
                    mel       = librosa.feature.melspectrogram(y=audio, sr=sr)

                    feat = np.concatenate([
                        mfccs.flatten(), zc.flatten(), sc.flatten(),
                        chroma.flatten(), bw.flatten(), contrast.flatten(),
                        rms.flatten(), rolloff.flatten(), tonnetz.flatten(),
                        mel.flatten()
                    ])
                    max_len = 3000
                    feat = np.pad(feat, (0, max(0, max_len - len(feat))))[:max_len]
                    X_list.append(feat)
                    Y_list.append(cats.index(os.path.basename(root)))
                    msg = f"✓ {fname}"
                    self.after(0, lambda m=msg: self.log(m, "dim"))
                except Exception as e:
                    self.after(0, lambda m=f"✗ {fname}: {e}": self.log(m, "err"))

        if not X_list:
            self.after(0, lambda: self.log("No features extracted.", "err"))
            return

        app_state["X"] = np.array(X_list)
        app_state["Y"] = np.array(Y_list)
        np.save(X_file, app_state["X"])
        np.save(Y_file, app_state["Y"])
        n = len(X_list)
        self.after(0, lambda: self.log(f"Saved {n} feature vectors.", "ok"))
        self.after(0, lambda: self._mark_step_done(1))

    # 3. Split ────────────────────────────────────────────────
    def cmd_split(self):
        self._run_in_thread(self._split_task)

    def _split_task(self):
        self.after(0, lambda: self.log_head("DATA SPLITTING"))

        if app_state["X"] is None:
            if os.path.exists(X_file) and os.path.exists(Y_file):
                app_state["X"] = np.load(X_file, allow_pickle=True)
                app_state["Y"] = np.load(Y_file, allow_pickle=True)
            else:
                self.after(0, lambda: self.log("Run MFCC extraction first.", "err"))
                return

        X, Y = app_state["X"], app_state["Y"]
        (app_state["X_train"], app_state["X_test"],
         app_state["y_train"], app_state["y_test"]) = train_test_split(
            X, Y, test_size=0.2, random_state=42)

        tr, te = len(app_state["X_train"]), len(app_state["X_test"])
        self.after(0, lambda: self.log(f"Train: {tr}  |  Test: {te}", "ok"))
        self.after(0, lambda: self._mark_step_done(2))

    # 4. Logistic ─────────────────────────────────────────────
    def cmd_train_logistic(self):
        self._run_in_thread(self._logistic_task)

    def _logistic_task(self):
        self.after(0, lambda: self.log_head("LOGISTIC REGRESSION"))
        if app_state["X_train"] is None:
            self.after(0, lambda: self.log("Run Data Split first.", "err"))
            return

        Xt, Xe = app_state["X_train"], app_state["X_test"]
        yt, ye = app_state["y_train"], app_state["y_test"]

        if os.path.exists(logreg_path):
            mdl = joblib.load(logreg_path)
            self.after(0, lambda: self.log("Loaded saved model.", "info"))
        else:
            self.after(0, lambda: self.log("Training…", "info"))
            mdl = LogisticRegression(max_iter=500)
            mdl.fit(Xt, yt)
            joblib.dump(mdl, logreg_path)
            self.after(0, lambda: self.log("Model saved.", "ok"))

        yp   = mdl.predict(Xe)
        acc  = accuracy_score(ye, yp)*100
        prec = precision_score(ye, yp, average="macro", zero_division=0)*100
        rec  = recall_score(ye, yp, average="macro", zero_division=0)*100
        f1   = f1_score(ye, yp, average="macro", zero_division=0)*100

        self.after(0, lambda: self._update_metrics(acc, prec, rec, f1))
        self.after(0, lambda: self.log(
            f"Accuracy {acc:.2f}%  Precision {prec:.2f}%  Recall {rec:.2f}%  F1 {f1:.2f}%", "ok"))
        self.after(0, lambda: self._mark_step_done(3))
        self.after(0, lambda: self._show_results_popup(ye, yp, mdl, Xe, "Logistic Regression"))

    # 5. LDA ──────────────────────────────────────────────────
    def cmd_train_lda(self):
        self._run_in_thread(self._lda_task)

    def _lda_task(self):
        self.after(0, lambda: self.log_head("LINEAR DISCRIMINANT ANALYSIS"))
        if app_state["X_train"] is None:
            self.after(0, lambda: self.log("Run Data Split first.", "err"))
            return

        Xt, Xe = app_state["X_train"], app_state["X_test"]
        yt, ye = app_state["y_train"], app_state["y_test"]

        if os.path.exists(lda_path):
            mdl = joblib.load(lda_path)
            self.after(0, lambda: self.log("Loaded saved LDA model.", "info"))
        else:
            self.after(0, lambda: self.log("Training…", "info"))
            mdl = LinearDiscriminantAnalysis()
            mdl.fit(Xt, yt)
            joblib.dump(mdl, lda_path)
            self.after(0, lambda: self.log("Model saved.", "ok"))

        yp   = mdl.predict(Xe)
        acc  = accuracy_score(ye, yp)*100
        prec = precision_score(ye, yp, average="macro", zero_division=0)*100
        rec  = recall_score(ye, yp, average="macro", zero_division=0)*100
        f1   = f1_score(ye, yp, average="macro", zero_division=0)*100

        self.after(0, lambda: self._update_metrics(acc, prec, rec, f1))
        self.after(0, lambda: self.log(
            f"Accuracy {acc:.2f}%  Precision {prec:.2f}%  Recall {rec:.2f}%  F1 {f1:.2f}%", "ok"))
        self.after(0, lambda: self._mark_step_done(4))
        self.after(0, lambda: self._show_results_popup(ye, yp, mdl, Xe, "LDA"))

    # 6. HuBERT + TAO ─────────────────────────────────────────
    def cmd_train_hubert(self):
        self._run_in_thread(self._hubert_task)

    def _hubert_task(self):
        self.after(0, lambda: self.log_head("HuBERT + TAO TREE CLASSIFIER"))

        if os.path.exists(X_hub_file) and os.path.exists(Y_hub_file):
            app_state["X_hubert"] = np.load(X_hub_file)
            app_state["Y_hubert"] = np.load(Y_hub_file)
            self.after(0, lambda: self.log("Loaded cached HuBERT embeddings.", "ok"))
        else:
            path = app_state["dataset_path"]
            cats = app_state["categories"]
            if not path:
                self.after(0, lambda: self.log("Upload dataset first.", "err"))
                return
            try:
                import torch
                from transformers import Wav2Vec2FeatureExtractor, HubertModel
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                fe   = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_NAME)
                hub  = HubertModel.from_pretrained(HUBERT_NAME).to(device)
                hub.eval()
                self.after(0, lambda: self.log("HuBERT model loaded.", "ok"))
            except Exception as e:
                self.after(0, lambda: self.log(f"HuBERT load failed: {e}", "err"))
                return

            embs, lbls = [], []
            for root, dirs, files in os.walk(path):
                for fname in files:
                    if not fname.lower().endswith(".wav") or "Thumbs.db" in fname:
                        continue
                    fp = os.path.join(root, fname)
                    try:
                        wav, sr = librosa.load(fp, sr=None)
                        if sr != 16000:
                            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                        inp = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
                        with torch.no_grad():
                            out = hub(inp).last_hidden_state
                            emb = out.mean(dim=1).squeeze().cpu().numpy()
                        embs.append(emb)
                        lbls.append(cats.index(os.path.basename(root)))
                        msg = f"✓ {fname}"
                        self.after(0, lambda m=msg: self.log(m, "dim"))
                    except Exception as e:
                        self.after(0, lambda m=f"✗ {fname}: {e}": self.log(m, "err"))

            if not embs:
                self.after(0, lambda: self.log("No embeddings created.", "err"))
                return
            app_state["X_hubert"] = np.vstack(embs)
            app_state["Y_hubert"] = np.array(lbls)
            np.save(X_hub_file, app_state["X_hubert"])
            np.save(Y_hub_file, app_state["Y_hubert"])
            self.after(0, lambda: self.log(f"Saved {len(embs)} HuBERT embeddings.", "ok"))

        Xh, Yh = app_state["X_hubert"], app_state["Y_hubert"]
        Xt, Xe, yt, ye = train_test_split(Xh, Yh, test_size=0.2,
                                           random_state=42, stratify=Yh)

        if os.path.exists(Tao_hub_path):
            mdl = joblib.load(Tao_hub_path)
            self.after(0, lambda: self.log("Loaded saved TAO model.", "info"))
        else:
            try:
                from TAO_TreeClassifier import TAO_TreeClassifier
                self.after(0, lambda: self.log("Training TAO…", "info"))
                mdl = TAO_TreeClassifier()
                mdl.fit(Xt, yt)
                joblib.dump(mdl, Tao_hub_path)
                self.after(0, lambda: self.log("TAO model saved.", "ok"))
                app_state["Tao_hubert"] = mdl
            except ImportError:
                self.after(0, lambda: self.log("TAO_TreeClassifier not found – using RandomForest fallback.", "warn"))
                from sklearn.ensemble import RandomForestClassifier
                mdl = RandomForestClassifier(n_estimators=200, random_state=42)
                mdl.fit(Xt, yt)
                joblib.dump(mdl, Tao_hub_path)
                app_state["Tao_hubert"] = mdl

        yp   = mdl.predict(Xe)
        acc  = accuracy_score(ye, yp)*100
        prec = precision_score(ye, yp, average="macro", zero_division=0)*100
        rec  = recall_score(ye, yp, average="macro", zero_division=0)*100
        f1   = f1_score(ye, yp, average="macro", zero_division=0)*100

        self.after(0, lambda: self._update_metrics(acc, prec, rec, f1))
        self.after(0, lambda: self.log(
            f"Accuracy {acc:.2f}%  Precision {prec:.2f}%  Recall {rec:.2f}%  F1 {f1:.2f}%", "ok"))
        self.after(0, lambda: self._mark_step_done(5))
        self.after(0, lambda: self._show_results_popup(ye, yp, mdl, Xe, "HuBERT + TAO"))

    # 7. Predict ──────────────────────────────────────────────
    def cmd_predict(self):
        fp = filedialog.askopenfilename(
            title="Select WAV file to predict",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not fp:
            return
        self._run_in_thread(lambda: self._predict_task(fp))

    def _predict_task(self, fp):
        self.after(0, lambda: self.log_head("PREDICTION"))
        self.after(0, lambda: self.log(f"File: {os.path.basename(fp)}", "info"))

        # Load waveform for display
        try:
            audio_vis, sr_vis = librosa.load(fp, sr=None)
            self.after(0, lambda: self._plot_waveform(
                audio_vis, sr_vis, title=os.path.basename(fp)))
        except Exception as e:
            self.after(0, lambda: self.log(f"Waveform load failed: {e}", "err"))

        # Try HuBERT+TAO first
        if os.path.exists(Tao_hub_path):
            try:
                import torch
                from transformers import Wav2Vec2FeatureExtractor, HubertModel
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                fe  = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_NAME)
                hub = HubertModel.from_pretrained(HUBERT_NAME).to(device)
                hub.eval()
                mdl = joblib.load(Tao_hub_path)
                wav, sr = librosa.load(fp, sr=None)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                inp = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"].to(device)
                with torch.no_grad():
                    out = hub(inp).last_hidden_state
                    emb = out.mean(dim=1).squeeze().cpu().numpy().reshape(1, -1)
                pred  = mdl.predict(emb)[0]
                label = LABELS[pred] if pred < len(LABELS) else str(pred)
                self.after(0, lambda: self.log(f"Model      : HuBERT + TAO", "dim"))
                self.after(0, lambda: self.log(f"Prediction : {label}", "ok"))
                self.after(0, lambda: self._show_prediction_popup(label, os.path.basename(fp)))
                return
            except Exception as e:
                self.after(0, lambda: self.log(f"HuBERT+TAO predict failed: {e}", "err"))

        # Fallback: logistic / LDA on MFCC
        for model_path, name in [(logreg_path, "Logistic"), (lda_path, "LDA")]:
            if os.path.exists(model_path):
                try:
                    mdl   = joblib.load(model_path)
                    audio, sr = librosa.load(fp, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).flatten()
                    max_l = 3000
                    feat  = np.pad(mfccs, (0, max(0, max_l - len(mfccs))))[:max_l].reshape(1,-1)
                    pred  = mdl.predict(feat)[0]
                    label = LABELS[int(pred)] if int(pred) < len(LABELS) else str(pred)
                    self.after(0, lambda: self.log(f"Model      : {name}", "dim"))
                    self.after(0, lambda: self.log(f"Prediction : {label}", "ok"))
                    self.after(0, lambda: self._show_prediction_popup(label, os.path.basename(fp)))
                    return
                except Exception as e:
                    self.after(0, lambda: self.log(f"{name} predict failed: {e}", "err"))

        self.after(0, lambda: self.log("No trained model found. Train a model first.", "err"))

    # ── Show charts ───────────────────────────────────────────
    def cmd_show_charts(self):
        if app_state["Y"] is None and os.path.exists(Y_file):
            app_state["Y"] = np.load(Y_file, allow_pickle=True)
        if app_state["Y"] is None:
            self.log("No Y data available. Run MFCC extraction first.", "err")
            return
        win = tk.Toplevel(self)
        win.title("Class Distribution")
        win.configure(bg="white")
        fig = Figure(figsize=(7, 4.5), facecolor="white")
        ax  = fig.add_subplot(111, facecolor="white")
        cats   = app_state["categories"] or [str(i) for i in range(len(np.unique(app_state["Y"])))]
        unique, counts = np.unique(app_state["Y"], return_counts=True)
        ax.bar([cats[i] if i < len(cats) else str(i) for i in unique],
               counts, color=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"][:len(unique)],
               edgecolor="white", width=0.6)
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333", labelsize=9)
        ax.set_title("Class Distribution", color="#111111", fontsize=12, fontweight="bold")
        ax.set_xlabel("Class", color="#444444", fontsize=9)
        ax.set_ylabel("Count",  color="#444444", fontsize=9)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        fig.tight_layout(pad=1.2)
        c = FigureCanvasTkAgg(fig, master=win)
        c.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ── Combined side-by-side results popup ───────────────────
    def _show_results_popup(self, y_true, y_pred, mdl, X_test, title):
        """Confusion Matrix (left) and ROC Curve (right) — white background."""
        win = tk.Toplevel(self)
        win.title(f"Results — {title}")
        win.configure(bg="white")
        win.geometry("1120x520")

        # Title bar
        hdr = tk.Frame(win, bg="#1a1a2e", height=42)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  {title}  ·  Evaluation Results",
                 bg="#1a1a2e", fg="white",
                 font=("Consolas", 11, "bold")).pack(side="left", padx=14, pady=10)
        tk.Button(hdr, text="  ✕  Close  ", bg="#c0392b", fg="white",
                  font=("Consolas", 9, "bold"), relief="flat",
                  cursor="hand2", command=win.destroy).pack(side="right", padx=12, pady=8)

        body = tk.Frame(win, bg="white")
        body.pack(fill="both", expand=True)

        # ── LEFT: Confusion Matrix ────────────────────────────
        left_f = tk.Frame(body, bg="white",
                          highlightbackground="#dddddd", highlightthickness=1)
        left_f.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)
        tk.Label(left_f, text="Confusion Matrix", bg="white",
                 fg="#111111", font=("Consolas", 11, "bold")).pack(pady=(10, 2))

        cm  = confusion_matrix(y_true, y_pred)
        lbl = LABELS[:cm.shape[0]]
        fig_cm = Figure(figsize=(5, 4), facecolor="white")
        ax_cm  = fig_cm.add_subplot(111, facecolor="white")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=lbl, yticklabels=lbl,
                    linewidths=0.8, linecolor="#eeeeee",
                    annot_kws={"size": 11, "weight": "bold", "color": "#111111"})
        ax_cm.set_xlabel("Predicted Label", fontsize=9, color="#444444")
        ax_cm.set_ylabel("True Label",      fontsize=9, color="#444444")
        ax_cm.tick_params(colors="#333333", labelsize=8)
        try:
            cbar = ax_cm.collections[0].colorbar
            cbar.ax.tick_params(colors="#333333", labelsize=7)
        except Exception:
            pass
        fig_cm.tight_layout(pad=1.0)
        FigureCanvasTkAgg(fig_cm, master=left_f).get_tk_widget().pack(
            fill="both", expand=True, padx=8, pady=(0, 10))

        # ── RIGHT: ROC Curve ──────────────────────────────────
        right_f = tk.Frame(body, bg="white",
                           highlightbackground="#dddddd", highlightthickness=1)
        right_f.pack(side="left", fill="both", expand=True, padx=(6, 12), pady=12)
        tk.Label(right_f, text="ROC Curves (per class)", bg="white",
                 fg="#111111", font=("Consolas", 11, "bold")).pack(pady=(10, 2))

        fig_roc = Figure(figsize=(5, 4), facecolor="white")
        ax_roc  = fig_roc.add_subplot(111, facecolor="white")
        roc_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

        if hasattr(mdl, "predict_proba"):
            try:
                y_score  = mdl.predict_proba(X_test)
                classes  = np.arange(y_score.shape[1])
                y_test_b = label_binarize(y_true, classes=classes)
                for i, color in zip(range(y_score.shape[1]), roc_colors):
                    fpr, tpr, _ = roc_curve(y_test_b[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    lbl_name = LABELS[i] if i < len(LABELS) else str(i)
                    ax_roc.plot(fpr, tpr, color=color, linewidth=2.2,
                                label=f"{lbl_name}  AUC={roc_auc:.2f}")
                ax_roc.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1)
                ax_roc.legend(fontsize=7.5, framealpha=0.95,
                              facecolor="white", edgecolor="#cccccc",
                              labelcolor="#222222", loc="lower right")
            except Exception as e:
                ax_roc.text(0.5, 0.5, f"ROC unavailable:\n{e}",
                            ha="center", va="center", color="#888888",
                            transform=ax_roc.transAxes, fontsize=9)
        else:
            ax_roc.text(0.5, 0.5, "Model does not\nsupport predict_proba",
                        ha="center", va="center", color="#888888",
                        transform=ax_roc.transAxes, fontsize=10)

        ax_roc.set_xlabel("False Positive Rate", fontsize=9, color="#444444")
        ax_roc.set_ylabel("True Positive Rate",  fontsize=9, color="#444444")
        ax_roc.tick_params(colors="#333333", labelsize=8)
        for sp in ax_roc.spines.values():
            sp.set_color("#cccccc")
        fig_roc.tight_layout(pad=1.0)
        FigureCanvasTkAgg(fig_roc, master=right_f).get_tk_widget().pack(
            fill="both", expand=True, padx=8, pady=(0, 10))

    # Old stub wrappers — now call combined popup
    def _show_cm_popup(self, y_true, y_pred, title):
        pass  # handled by _show_results_popup

    def _show_roc_popup(self, mdl, X_test, y_test, title):
        pass  # handled by _show_results_popup


    def _show_prediction_popup(self, label, filename):
        color_map = {
            "Air leak":         CLR["warn"],
            "Oil leak":         CLR["warn"],
            "Background noise": CLR["yellow"],
            "Idling":           CLR["yellow"],
            "Normal":           CLR["accent2"],
        }
        color = color_map.get(label, CLR["accent"])
        icon  = "\u26a0" if "leak" in label.lower() else ("\u2714" if label == "Normal" else "\u25c8")

        W, H = 460, 300
        win = tk.Toplevel(self)
        win.title("Prediction Result")
        win.configure(bg=CLR["bg"])
        win.resizable(False, False)

        # Centre on screen
        win.update_idletasks()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x  = (sw - W) // 2
        y  = (sh - H) // 2
        win.geometry(f"{W}x{H}+{x}+{y}")

        # Coloured accent strip at top
        tk.Frame(win, bg=color, height=5).pack(fill="x")

        tk.Label(win, text="PREDICTION RESULT",
                 bg=CLR["bg"], fg=CLR["text_dim"],
                 font=("Consolas", 9, "bold")).pack(pady=(22, 2))

        tk.Label(win, text=filename,
                 bg=CLR["bg"], fg=CLR["text_dim"],
                 font=("Consolas", 9)).pack()

        tk.Frame(win, bg=CLR["border"], height=1).pack(fill="x", padx=40, pady=14)

        # Icon + label side by side, centred
        row = tk.Frame(win, bg=CLR["bg"])
        row.pack()
        tk.Label(row, text=icon, bg=CLR["bg"], fg=color,
                 font=("Consolas", 32)).pack(side="left", padx=(0, 12))
        tk.Label(row, text=label.upper(), bg=CLR["bg"], fg=color,
                 font=("Consolas", 24, "bold")).pack(side="left")

        tk.Frame(win, bg=CLR["border"], height=1).pack(fill="x", padx=40, pady=14)

        tk.Button(win, text="  Close  ",
                  bg=color, fg="#000000" if label == "Normal" else "#ffffff",
                  activebackground=CLR["border"],
                  font=("Consolas", 10, "bold"),
                  relief="flat", cursor="hand2",
                  command=win.destroy).pack(pady=(0, 18))


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
