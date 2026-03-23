"""
Scoliosis AI – Glassmorphism HUD Launcher
Dark-glass futuristic UI inspired by Stark Industries
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
from pathlib import Path
import sys
import os


# ═══════════════════════════════════════════════════════════════════════
#  COLOUR SYSTEM — dark-glass palette
# ═══════════════════════════════════════════════════════════════════════
class HUD:
    BG          = "#0a0e17"       # deep space black
    PANEL       = "#111827"       # raised card
    GLASS       = "#1a2234"       # frosted glass mid-tone
    GLASS_LT    = "#1f2b3e"       # lighter glass for inputs
    BORDER      = "#2d3a50"       # subtle borders
    ACCENT      = "#00e5ff"       # cyan neon (primary accent)
    ACCENT_DIM  = "#0097a7"       # dimmed accent
    SUCCESS     = "#00e676"       # neon green
    WARNING     = "#ffab00"       # amber
    DANGER      = "#ff1744"       # red alert
    TEXT        = "#e0e6ed"       # primary text
    TEXT_DIM    = "#7b8a9e"       # muted text
    TEXT_BRIGHT = "#ffffff"       # high-contrast text
    GLOW        = "#00e5ff"       # glow colour = accent
    HEADER_BG   = "#060a12"


# ═══════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════
class ScoliosisAISimple:
    """Glassmorphism HUD launcher for Scoliosis AI"""

    def __init__(self, root):
        self.root = root
        self.root.title("S C O L I O S I S   A I  —  D I A G N O S T I C   S Y S T E M")
        self.root.geometry("1060x740")
        self.root.resizable(True, True)
        self.root.minsize(900, 620)
        self.root.configure(bg=HUD.BG)

        # Variables
        self.selected_image = tk.StringVar()
        self.confidence = tk.DoubleVar(value=0.25)
        self.analysis_mode = tk.StringVar(value="scoliosis_enhanced")
        self.report_model = tk.StringVar(value="template")
        self.model_type = tk.StringVar(value="all")
        self.auto_open_image = tk.BooleanVar(value=True)
        self.is_running = False

        self.mode_config = {
            "yolov8n_base": {
                "label": "YOLOv8n Base",
                "yolo_model": "models/detection/yolov8n.pt",
                "model_type": "yolo",
                "confidence": 0.30,
                "note": "Fastest inference · baseline accuracy"
            },
            "scoliosis_enhanced": {
                "label": "Scoliosis YOLO Enhanced",
                "yolo_model": "models/detection/scoliosis_yolo_enhanced/weights/best.pt",
                "model_type": "all",
                "confidence": 0.25,
                "note": "Recommended · balanced accuracy"
            },
            "scoliosis_high_accuracy": {
                "label": "Scoliosis YOLO High Accuracy",
                "yolo_model": "models/detection/scoliosis_yolo_high_accuracy/weights/best.pt",
                "model_type": "all",
                "confidence": 0.20,
                "note": "Best accuracy · dual-model detection"
            }
        }

        self._setup_styles()
        self._build_ui()

    # ── ttk styling ─────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame", background=HUD.BG)
        s.configure("Glass.TFrame", background=HUD.GLASS)
        s.configure("TLabel", background=HUD.BG, foreground=HUD.TEXT, font=("Segoe UI", 10))
        s.configure("Dim.TLabel", background=HUD.GLASS, foreground=HUD.TEXT_DIM, font=("Segoe UI", 9))
        s.configure("TLabelframe", background=HUD.GLASS, bordercolor=HUD.BORDER, relief="solid", borderwidth=1)
        s.configure("TLabelframe.Label", background=HUD.GLASS, foreground=HUD.ACCENT, font=("Segoe UI Semibold", 10))
        s.configure("TCombobox", fieldbackground=HUD.GLASS_LT, background=HUD.GLASS_LT,
                     foreground=HUD.TEXT, borderwidth=1, arrowsize=14, padding=5)
        s.map("TCombobox", fieldbackground=[("readonly", HUD.GLASS_LT)])
        s.configure("Cyan.Horizontal.TProgressbar", troughcolor=HUD.PANEL, background=HUD.ACCENT, thickness=6)
        s.configure("Horizontal.TScale", troughcolor=HUD.PANEL, background=HUD.ACCENT)

    # ── hover helper ────────────────────────────────────────────────────
    def _hover(self, w, normal, hover, fg_n=None, fg_h=None):
        def enter(_):
            w.config(bg=hover)
            if fg_h:
                w.config(fg=fg_h)
        def leave(_):
            w.config(bg=normal)
            if fg_n:
                w.config(fg=fg_n)
        w.bind("<Enter>", enter)
        w.bind("<Leave>", leave)

    # ── glow-border card factory ────────────────────────────────────────
    def _glass_card(self, parent, glow=HUD.BORDER, **kw):
        outer = tk.Frame(parent, bg=glow, bd=0, highlightthickness=0)
        inner = tk.Frame(outer, bg=HUD.GLASS, bd=0, highlightthickness=0, **kw)
        inner.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)
        return outer, inner

    # ── neon button factory ─────────────────────────────────────────────
    def _neon_btn(self, parent, text, cmd, color=HUD.ACCENT, width=None, big=False):
        font = ("Segoe UI Semibold", 13 if big else 10)
        btn = tk.Button(parent, text=text, command=cmd, bg=color, fg=HUD.BG,
                        activebackground=HUD.TEXT_BRIGHT, activeforeground=HUD.BG,
                        font=font, relief=tk.FLAT, cursor="hand2",
                        padx=20, pady=12 if big else 8,
                        **({"width": width} if width else {}))
        hover_bg = HUD.ACCENT_DIM if color == HUD.ACCENT else color
        self._hover(btn, color, hover_bg)
        return btn

    # ═══════════════════════════════════════════════════════════════════
    #  BUILD UI
    # ═══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── HEADER ──────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=HUD.HEADER_BG, height=100)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        # Accent glow line
        tk.Frame(hdr, bg=HUD.ACCENT, height=2).pack(fill=tk.X, side=tk.TOP)

        # Title row
        title_row = tk.Frame(hdr, bg=HUD.HEADER_BG)
        title_row.pack(fill=tk.X, pady=(16, 0))

        tk.Label(title_row, text="\u25c9", font=("Segoe UI", 22), bg=HUD.HEADER_BG,
                 fg=HUD.ACCENT).pack(side=tk.LEFT, padx=(24, 8))
        tk.Label(title_row, text="SCOLIOSIS AI", font=("Segoe UI Semibold", 26),
                 bg=HUD.HEADER_BG, fg=HUD.TEXT_BRIGHT).pack(side=tk.LEFT)
        tk.Label(title_row, text="DIAGNOSTIC SYSTEM v2.0", font=("Segoe UI", 11),
                 bg=HUD.HEADER_BG, fg=HUD.ACCENT_DIM).pack(side=tk.LEFT, padx=(12, 0), pady=(6, 0))

        # Subtitle
        tk.Label(hdr, text="Upload  \u2022  Detect  \u2022  Analyze  \u2022  Report",
                 font=("Segoe UI", 10), bg=HUD.HEADER_BG, fg=HUD.TEXT_DIM).pack(pady=(2, 0))

        # Bottom glow
        tk.Frame(hdr, bg=HUD.BORDER, height=1).pack(fill=tk.X, side=tk.BOTTOM)

        # ── BODY ────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=HUD.BG, padx=20, pady=14)
        body.pack(fill=tk.BOTH, expand=True)

        # ── Left column (controls) ─────────────────────────────────────
        left = tk.Frame(body, bg=HUD.BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Card 1 — Image selection
        c1_out, c1 = self._glass_card(left)
        c1_out.pack(fill=tk.X, pady=(0, 10))
        tk.Label(c1, text="\u25b8  SELECT X-RAY IMAGE", font=("Segoe UI Semibold", 10),
                 bg=HUD.GLASS, fg=HUD.ACCENT).pack(anchor=tk.W, padx=14, pady=(12, 6))

        row1 = tk.Frame(c1, bg=HUD.GLASS)
        row1.pack(fill=tk.X, padx=14, pady=(0, 12))
        img_entry = tk.Entry(row1, textvariable=self.selected_image, font=("Segoe UI", 10),
                             bg=HUD.GLASS_LT, fg=HUD.TEXT, insertbackground=HUD.ACCENT,
                             relief=tk.FLAT, bd=0, highlightthickness=1,
                             highlightbackground=HUD.BORDER, highlightcolor=HUD.ACCENT)
        img_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7, padx=(0, 8))
        self._neon_btn(row1, "BROWSE", self.browse_image).pack(side=tk.LEFT)

        # Card 2 — Settings
        c2_out, c2 = self._glass_card(left)
        c2_out.pack(fill=tk.X, pady=(0, 10))
        tk.Label(c2, text="\u25b8  CONFIGURATION", font=("Segoe UI Semibold", 10),
                 bg=HUD.GLASS, fg=HUD.ACCENT).pack(anchor=tk.W, padx=14, pady=(12, 6))

        settings = tk.Frame(c2, bg=HUD.GLASS, padx=14)
        settings.pack(fill=tk.X, pady=(0, 12))

        # Model row
        r_model = tk.Frame(settings, bg=HUD.GLASS)
        r_model.pack(fill=tk.X, pady=4)
        tk.Label(r_model, text="MODEL", font=("Segoe UI Semibold", 9), bg=HUD.GLASS,
                 fg=HUD.TEXT_DIM, width=14, anchor=tk.W).pack(side=tk.LEFT)
        mode_cb = ttk.Combobox(r_model, width=30, state="readonly",
                               values=[c["label"] for c in self.mode_config.values()])
        mode_cb.current(1)
        mode_cb.pack(side=tk.LEFT, padx=(0, 10))
        mode_cb.bind("<<ComboboxSelected>>", lambda _: self._on_mode_change(mode_cb.get()))
        self.mode_note_var = tk.StringVar(value=self.mode_config["scoliosis_enhanced"]["note"])
        tk.Label(r_model, textvariable=self.mode_note_var, font=("Segoe UI", 9),
                 bg=HUD.GLASS, fg=HUD.ACCENT_DIM).pack(side=tk.LEFT)

        # Report engine row
        r_rpt = tk.Frame(settings, bg=HUD.GLASS)
        r_rpt.pack(fill=tk.X, pady=4)
        tk.Label(r_rpt, text="REPORT", font=("Segoe UI Semibold", 9), bg=HUD.GLASS,
                 fg=HUD.TEXT_DIM, width=14, anchor=tk.W).pack(side=tk.LEFT)
        rpt_cb = ttk.Combobox(r_rpt, width=22, state="readonly",
                              values=["Template (Fast)", "Gemma 2B (Detailed)"])
        rpt_cb.current(0)
        rpt_cb.pack(side=tk.LEFT, padx=(0, 10))
        rpt_cb.bind("<<ComboboxSelected>>", lambda _: self._on_report_engine_change(rpt_cb.get()))
        tk.Label(r_rpt, text="Gemma gives richer clinical text", font=("Segoe UI", 9),
                 bg=HUD.GLASS, fg=HUD.TEXT_DIM).pack(side=tk.LEFT)

        # Confidence row
        r_conf = tk.Frame(settings, bg=HUD.GLASS)
        r_conf.pack(fill=tk.X, pady=4)
        tk.Label(r_conf, text="CONFIDENCE", font=("Segoe UI Semibold", 9), bg=HUD.GLASS,
                 fg=HUD.TEXT_DIM, width=14, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Scale(r_conf, from_=0.1, to=0.9, variable=self.confidence,
                  orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(r_conf, textvariable=self.confidence, font=("Segoe UI Semibold", 10),
                 bg=HUD.GLASS, fg=HUD.ACCENT, width=5).pack(side=tk.LEFT)

        # Auto-open checkbox
        r_auto = tk.Frame(settings, bg=HUD.GLASS)
        r_auto.pack(fill=tk.X, pady=4)
        tk.Checkbutton(r_auto, text="Auto-open annotated image after analysis",
                       variable=self.auto_open_image, bg=HUD.GLASS, fg=HUD.TEXT,
                       activebackground=HUD.GLASS, selectcolor=HUD.PANEL,
                       font=("Segoe UI", 9)).pack(side=tk.LEFT)

        # ── ANALYZE button ──────────────────────────────────────────────
        btn_frame = tk.Frame(left, bg=HUD.BG)
        btn_frame.pack(fill=tk.X, pady=(4, 10))
        self.analyze_btn = self._neon_btn(btn_frame, "\u25b6  ANALYZE X-RAY",
                                          self.run_diagnosis, color=HUD.SUCCESS, big=True)
        self.analyze_btn.pack(fill=tk.X)

        # Progress
        self.progress = ttk.Progressbar(left, mode="indeterminate", length=600,
                                        style="Cyan.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=(0, 8))

        # ── Output log ──────────────────────────────────────────────────
        log_out, log_inner = self._glass_card(left)
        log_out.pack(fill=tk.BOTH, expand=True)
        tk.Label(log_inner, text="\u25b8  ANALYSIS OUTPUT", font=("Segoe UI Semibold", 10),
                 bg=HUD.GLASS, fg=HUD.ACCENT).pack(anchor=tk.W, padx=14, pady=(10, 4))
        self.output_log = scrolledtext.ScrolledText(
            log_inner, height=10, font=("Cascadia Mono", 9),
            bg="#060b14", fg=HUD.ACCENT, insertbackground=HUD.ACCENT,
            relief=tk.FLAT, bd=0, highlightthickness=0, wrap=tk.WORD)
        self.output_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # ── Right column (action buttons) ──────────────────────────────
        right = tk.Frame(body, bg=HUD.BG, width=180)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="ACTIONS", font=("Segoe UI Semibold", 10),
                 bg=HUD.BG, fg=HUD.ACCENT).pack(pady=(0, 10))

        for label, cmd, color in [
            ("\U0001f4c1 Results Folder", lambda: self.open_folder("outputs/diagnosis"), HUD.ACCENT),
            ("\U0001f4c4 View Report", self.view_report, HUD.WARNING),
            ("\u2753 Help", self.show_help, HUD.TEXT_DIM),
        ]:
            b = self._neon_btn(right, label, cmd, color=color)
            b.pack(fill=tk.X, pady=4)

        # Spacer
        tk.Frame(right, bg=HUD.BG).pack(fill=tk.BOTH, expand=True)

        # System info in right column
        info_frame = tk.Frame(right, bg=HUD.PANEL)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        for lbl in ["ENGINE  YOLOv8", "MODE    Dual-Model", "RENDER  HUD v2.0", "PYTHON  " + sys.version.split()[0]]:
            tk.Label(info_frame, text=lbl, font=("Cascadia Mono", 8), bg=HUD.PANEL,
                     fg=HUD.TEXT_DIM, anchor=tk.W).pack(fill=tk.X, padx=8, pady=1)

        # ── STATUS BAR ─────────────────────────────────────────────────
        status = tk.Frame(self.root, bg=HUD.HEADER_BG, height=32)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        status.pack_propagate(False)
        tk.Frame(status, bg=HUD.ACCENT, height=1).pack(fill=tk.X, side=tk.TOP)

        self.status_var = tk.StringVar(value="\u25c9  READY  \u2022  Scoliosis YOLO Enhanced  \u2022  Template report")
        tk.Label(status, textvariable=self.status_var, bg=HUD.HEADER_BG, fg=HUD.ACCENT,
                 font=("Cascadia Mono", 9), anchor=tk.W, padx=16).pack(fill=tk.X, pady=6)
    
    # ═══════════════════════════════════════════════════════════════════
    #  EVENT HANDLERS & BUSINESS LOGIC  (preserved from original)
    # ═══════════════════════════════════════════════════════════════════
    def _on_mode_change(self, selected_label):
        for mode_key, cfg in self.mode_config.items():
            if cfg['label'] == selected_label:
                self.analysis_mode.set(mode_key)
                self.model_type.set(cfg['model_type'])
                self.confidence.set(cfg['confidence'])
                self.mode_note_var.set(cfg['note'])
                self.status_var.set(f"\u25c9  {cfg['label']}")
                break

    def _on_report_engine_change(self, selected_label):
        self.report_model.set('gemma' if 'Gemma' in selected_label else 'template')
        engine = "Gemma 2B" if self.report_model.get() == 'gemma' else "Template"
        self.status_var.set(f"\u25c9  Report engine: {engine}")

    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.selected_image.set(filename)
            self.status_var.set(f"\u25c9  {Path(filename).name}")
            self.output_log.insert(tk.END, f"[+] Image loaded: {Path(filename).name}\n\n")

    def run_diagnosis(self):
        if self.is_running:
            return
        if not self.selected_image.get():
            messagebox.showwarning("No Image", "Select an X-ray image first.")
            return
        if not Path(self.selected_image.get()).exists():
            messagebox.showerror("File Not Found", f"Image not found:\n{self.selected_image.get()}")
            return

        mode_cfg = self.mode_config[self.analysis_mode.get()]
        self.is_running = True
        self.analyze_btn.config(state='disabled', text="\u23f3  ANALYZING ...")
        self.progress.start()
        self.status_var.set("\u25c9  Running AI analysis ...")
        self.output_log.delete(1.0, tk.END)
        self.output_log.insert(tk.END, "--- SCOLIOSIS AI DIAGNOSTIC ---\n")
        self.output_log.insert(tk.END, f"  Image    : {Path(self.selected_image.get()).name}\n")
        self.output_log.insert(tk.END, f"  Model    : {mode_cfg['label']}\n")
        self.output_log.insert(tk.END, f"  Report   : {'Gemma 2B' if self.report_model.get() == 'gemma' else 'Template'}\n")
        self.output_log.insert(tk.END, f"  Detector : {mode_cfg['yolo_model']}\n")
        self.output_log.insert(tk.END, f"  AutoOpen : {'ON' if self.auto_open_image.get() else 'OFF'}\n")
        self.output_log.insert(tk.END, f"  Conf     : {self.confidence.get():.2f}\n\n")

        thread = threading.Thread(target=self._run_diagnosis_thread)
        thread.daemon = True
        thread.start()

    def _run_diagnosis_thread(self):
        try:
            mode_cfg = self.mode_config[self.analysis_mode.get()]
            cmd = [
                sys.executable, "diagnose.py",
                "--image", self.selected_image.get(),
                "--model", self.model_type.get(),
                "--yolo-model", mode_cfg['yolo_model'],
                "--confidence", str(self.confidence.get()),
                "--report-model", self.report_model.get(),
                "--auto-open-image", 'true' if self.auto_open_image.get() else 'false',
            ]
            if self.report_model.get() == 'gemma':
                cmd.extend(["--gemma-model-name", "google/gemma-2b-it"])

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1
            )

            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                self.root.after(0, self._update_log, line)
            process.wait()

            if process.returncode == 0:
                self.root.after(0, self._diagnosis_complete)
            else:
                output_text = ''.join(output_lines) if output_lines else "Unknown error"
                self.root.after(0, self._diagnosis_error, output_text)
        except Exception as e:
            self.root.after(0, self._diagnosis_error, f"System error: {str(e)}")

    def _update_log(self, text):
        self.output_log.insert(tk.END, text)
        self.output_log.see(tk.END)

    def _diagnosis_complete(self):
        self.is_running = False
        self.progress.stop()
        self.analyze_btn.config(state='normal', text="\u25b6  ANALYZE X-RAY")
        self.status_var.set("\u25c9  ANALYSIS COMPLETE")

        self.output_log.insert(tk.END, "\n" + "\u2550" * 50 + "\n")
        self.output_log.insert(tk.END, "  DIAGNOSIS COMPLETE\n")
        self.output_log.insert(tk.END, "\u2550" * 50 + "\n\n")
        self.output_log.insert(tk.END, "  Results  \u2192  outputs/diagnosis/\n")
        self.output_log.insert(tk.END, "  Click 'View Report' for detailed results\n")

        messagebox.showinfo(
            "Analysis Complete",
            "Diagnosis finished successfully.\n\n"
            "Results saved to: outputs/diagnosis/\n\n"
            "Use the action buttons to view results."
        )

    def _diagnosis_error(self, error_output=None):
        self.is_running = False
        self.progress.stop()
        self.analyze_btn.config(state='normal', text="\u25b6  ANALYZE X-RAY")
        self.status_var.set("\u25c9  ANALYSIS FAILED")

        error_msg = "Analysis failed.\n\n"
        if error_output:
            lines = str(error_output).split('\n')
            for line in lines:
                if '\u274c Error:' in line:
                    error_msg += line.strip() + "\n"
                elif 'No spine' in line:
                    error_msg += "No spine detected.\n  - Is this an X-ray?\n  - Lower confidence threshold.\n"
                    break
                elif 'Multiple spines' in line:
                    error_msg += "Multiple spines detected.\n  - Use a single-patient X-ray.\n"
                    break
                elif 'Failed to load model' in line or 'Failed to run inference' in line:
                    error_msg += line.strip() + "\n"
            if error_msg.count('\n') <= 2:
                error_msg += "\nCheck log for details.\n\nTroubleshooting:\n"
                error_msg += "\u2022 Valid X-ray image (PNG/JPEG)\n"
                error_msg += "\u2022 Lower confidence (0.1-0.2)\n"
                error_msg += "\u2022 Models installed correctly\n"
        else:
            error_msg += "Unknown error. See log.\n"
        messagebox.showerror("Analysis Failed", error_msg)

    def open_folder(self, path):
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
            self.status_var.set(f"\u25c9  Opened: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")

    def view_report(self):
        try:
            reports = list(Path("outputs/diagnosis").glob("*_report.txt"))
            if reports:
                latest = max(reports, key=lambda p: p.stat().st_mtime)
                if sys.platform == 'win32':
                    os.startfile(latest)
                else:
                    subprocess.Popen(['open', latest] if sys.platform == 'darwin' else ['xdg-open', latest])
                self.status_var.set(f"\u25c9  {latest.name}")
            else:
                messagebox.showinfo("No Reports", "Run an analysis first to generate a report.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open report:\n{e}")

    def show_help(self):
        hw = tk.Toplevel(self.root)
        hw.title("SCOLIOSIS AI \u2014 HELP")
        hw.geometry("620x520")
        hw.configure(bg=HUD.BG)
        tk.Frame(hw, bg=HUD.ACCENT, height=2).pack(fill=tk.X)

        txt = scrolledtext.ScrolledText(hw, font=("Cascadia Mono", 9), wrap=tk.WORD,
                                        bg=HUD.PANEL, fg=HUD.TEXT, padx=20, pady=16,
                                        insertbackground=HUD.ACCENT, relief=tk.FLAT)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert(tk.END, HELP_TEXT.strip())
        txt.config(state='disabled')

        self._neon_btn(hw, "CLOSE", hw.destroy, color=HUD.ACCENT).pack(pady=10)


# ─────────────────────────────────────────────────────────────────────
HELP_TEXT = """
SCOLIOSIS AI — DIAGNOSTIC SYSTEM v2.0

HOW TO USE
  1.  Browse → select X-ray image
  2.  Choose detection model
  3.  Adjust confidence if needed (0.25 default)
  4.  Click ANALYZE X-RAY
  5.  View results in output log or click View Report

WHAT YOU GET
  • Spine detection with targeting brackets
  • Cobb angle estimation
  • Severity classification (Normal / Mild / Moderate / Severe)
  • Annotated image with HUD overlay
  • Detailed text report
  • JSON summary for integration

CONFIDENCE THRESHOLD
  0.25  Recommended (Enhanced model)
  Lower → more detections, less strict
  Higher → fewer detections, more strict

TIPS
  • Use clear, high-quality X-ray images
  • Spine must be visible in image
  • High Accuracy model gives best results
  • Base model is fastest
  • Gemma report mode generates richer text
"""


def main():
    root = tk.Tk()
    ScoliosisAISimple(root)
    root.mainloop()


if __name__ == "__main__":
    main()
