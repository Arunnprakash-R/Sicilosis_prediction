"""
Scoliosis AI - Simplified GUI (Diagnosis Focus)
Clean, simple interface for X-ray diagnosis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
from pathlib import Path
import sys
import os


class ModernStyle:
    """Modern color scheme"""
    PRIMARY = "#2c3e50"
    SUCCESS = "#27ae60"
    WARNING = "#f39c12"
    LIGHT = "#ecf0f1"
    DARK = "#2c2c2c"
    WHITE = "#ffffff"


class ScoliosisAISimple:
    """Simplified GUI - Diagnosis focused"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Scoliosis AI - X-ray Diagnosis System")
        self.root.geometry("950x650")
        self.root.resizable(True, True)
        self.root.minsize(800, 500)
        
        # Setup styles
        self.setup_styles()
        
        # Variables
        self.selected_image = tk.StringVar()
        self.confidence = tk.DoubleVar(value=0.25)
        
        # Create UI
        self.create_ui()
        
    def setup_styles(self):
        """Setup modern UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=ModernStyle.LIGHT)
        style.configure('TLabel', background=ModernStyle.LIGHT, foreground=ModernStyle.DARK)
        style.configure('TLabelframe', background=ModernStyle.LIGHT)
        style.configure('TLabelframe.Label', background=ModernStyle.LIGHT, foreground=ModernStyle.PRIMARY, font=('Arial', 10, 'bold'))
        
    def create_ui(self):
        """Create the user interface"""
        
        # Header
        header_frame = tk.Frame(self.root, bg=ModernStyle.PRIMARY, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🏥 Scoliosis AI - X-ray Diagnosis",
            font=("Arial", 20, "bold"),
            bg=ModernStyle.PRIMARY,
            fg=ModernStyle.WHITE
        )
        title_label.pack(pady=10)
        
        subtitle = tk.Label(
            header_frame,
            text="Upload X-ray • Analyze • Get Results",
            font=("Arial", 10),
            bg=ModernStyle.PRIMARY,
            fg=ModernStyle.LIGHT
        )
        subtitle.pack()
        
        # Main content
        main_frame = ttk.Frame(self.root, padding="25")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="📋 Select an X-ray image and click 'Analyze' to get instant AI diagnosis",
            font=("Arial", 11),
            fg=ModernStyle.PRIMARY,
            bg=ModernStyle.LIGHT
        )
        instructions.pack(pady=(0, 20))
        
        # Image selection
        image_frame = ttk.LabelFrame(main_frame, text="Step 1: Select X-ray Image", padding="20")
        image_frame.pack(fill=tk.X, pady=15)
        
        entry_frame = tk.Frame(image_frame, bg=ModernStyle.LIGHT)
        entry_frame.pack(fill=tk.X)
        
        image_entry = ttk.Entry(entry_frame, textvariable=self.selected_image, width=70, font=("Arial", 10))
        image_entry.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        tk.Button(
            entry_frame,
            text="📁 Browse",
            command=self.browse_image,
            bg=ModernStyle.PRIMARY,
            fg=ModernStyle.WHITE,
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=5)
        
        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Step 2: Adjust Settings (Optional)", padding="20")
        settings_frame.pack(fill=tk.X, pady=15)
        
        conf_frame = tk.Frame(settings_frame, bg=ModernStyle.LIGHT)
        conf_frame.pack(fill=tk.X)
        
        tk.Label(conf_frame, text="Detection Confidence:", font=("Arial", 10, "bold"), bg=ModernStyle.LIGHT).pack(side=tk.LEFT, padx=10)
        
        confidence_scale = ttk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            variable=self.confidence,
            orient=tk.HORIZONTAL,
            length=250
        )
        confidence_scale.pack(side=tk.LEFT, padx=10)
        
        confidence_label = tk.Label(
            conf_frame,
            textvariable=self.confidence,
            width=6,
            font=("Arial", 10, "bold"),
            bg=ModernStyle.LIGHT,
            fg=ModernStyle.SUCCESS
        )
        confidence_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(conf_frame, text="(0.25 = recommended)", font=("Arial", 9), fg="#666", bg=ModernStyle.LIGHT).pack(side=tk.LEFT)
        
        # Analyze button
        btn_frame = tk.Frame(main_frame, bg=ModernStyle.LIGHT)
        btn_frame.pack(pady=25)
        
        self.analyze_btn = tk.Button(
            btn_frame,
            text="🔬 ANALYZE X-RAY",
            command=self.run_diagnosis,
            bg=ModernStyle.SUCCESS,
            fg=ModernStyle.WHITE,
            font=("Arial", 14, "bold"),
            width=30,
            height=2,
            relief=tk.FLAT,
            cursor="hand2",
            activebackground="#229954"
        )
        self.analyze_btn.pack()
        
        # Progress
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=600)
        self.progress.pack(pady=10)
        
        # Output log
        log_frame = ttk.LabelFrame(main_frame, text="Analysis Output", padding="15")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        self.output_log = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        self.output_log.pack(fill=tk.BOTH, expand=True)
        
        # Results buttons
        result_frame = tk.Frame(main_frame, bg=ModernStyle.LIGHT)
        result_frame.pack(pady=10)
        
        tk.Button(
            result_frame,
            text="📂 Open Results Folder",
            command=lambda: self.open_folder("outputs/diagnosis"),
            bg="#3498db",
            fg=ModernStyle.WHITE,
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=15,
            pady=10,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            result_frame,
            text="📄 View Latest Report",
            command=self.view_report,
            bg=ModernStyle.WARNING,
            fg=ModernStyle.WHITE,
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=15,
            pady=10,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            result_frame,
            text="ℹ️ Help",
            command=self.show_help,
            bg="#95a5a6",
            fg=ModernStyle.WHITE,
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=15,
            pady=10,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=10)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg=ModernStyle.DARK, height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready to analyze X-rays")
        status_bar = tk.Label(
            status_frame,
            textvariable=self.status_var,
            bg=ModernStyle.DARK,
            fg=ModernStyle.LIGHT,
            anchor=tk.W,
            font=("Arial", 9),
            padx=15
        )
        status_bar.pack(fill=tk.X, pady=5)
    
    def browse_image(self):
        """Browse for X-ray image"""
        filename = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.selected_image.set(filename)
            self.status_var.set(f"Selected: {Path(filename).name}")
            self.output_log.insert(tk.END, f"✅ Image selected: {Path(filename).name}\n\n")
    
    def run_diagnosis(self):
        """Run diagnosis on selected image"""
        if not self.selected_image.get():
            messagebox.showwarning("No Image Selected", "Please select an X-ray image first!")
            return
        
        if not Path(self.selected_image.get()).exists():
            messagebox.showerror("File Not Found", f"Image not found:\n{self.selected_image.get()}")
            return
        
        # Disable button and start progress
        self.analyze_btn.config(state='disabled', text="⏳ ANALYZING...")
        self.progress.start()
        self.status_var.set("Running AI analysis...")
        self.output_log.delete(1.0, tk.END)
        self.output_log.insert(tk.END, "🔬 Starting AI analysis...\n")
        self.output_log.insert(tk.END, f"📁 Image: {Path(self.selected_image.get()).name}\n")
        self.output_log.insert(tk.END, f"🎯 Confidence threshold: {self.confidence.get():.2f}\n\n")
        
        # Run in thread
        thread = threading.Thread(target=self._run_diagnosis_thread)
        thread.daemon = True
        thread.start()
    
    def _run_diagnosis_thread(self):
        """Run diagnosis in background thread"""
        try:
            cmd = [
                sys.executable,
                "diagnose.py",
                "--image", self.selected_image.get(),
                "--model", "all",
                "--confidence", str(self.confidence.get())
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            
            # Read output
            for line in process.stdout:
                self.root.after(0, self._update_log, line)
            
            process.wait()
            
            if process.returncode == 0:
                self.root.after(0, self._diagnosis_complete)
            else:
                self.root.after(0, self._diagnosis_error)
                
        except Exception as e:
            self.root.after(0, self._diagnosis_error, str(e))
    
    def _update_log(self, text):
        """Update output log"""
        self.output_log.insert(tk.END, text)
        self.output_log.see(tk.END)
    
    def _diagnosis_complete(self):
        """Handle diagnosis completion"""
        self.progress.stop()
        self.analyze_btn.config(state='normal', text="🔬 ANALYZE X-RAY")
        self.status_var.set("✅ Analysis complete!")
        
        self.output_log.insert(tk.END, "\n" + "="*60 + "\n")
        self.output_log.insert(tk.END, "✅ DIAGNOSIS COMPLETE!\n")
        self.output_log.insert(tk.END, "="*60 + "\n\n")
        self.output_log.insert(tk.END, "📁 Results saved to: outputs/diagnosis/\n")
        self.output_log.insert(tk.END, "📄 Click 'View Latest Report' to see detailed results\n")
        
        messagebox.showinfo(
            "Success ✅",
            "Diagnosis completed successfully!\n\n"
            "📁 Results saved to: outputs/diagnosis/\n\n"
            "Click 'View Latest Report' or 'Open Results Folder' to see the results."
        )
    
    def _diagnosis_error(self, error=None):
        """Handle diagnosis error"""
        self.progress.stop()
        self.analyze_btn.config(state='normal', text="🔬 ANALYZE X-RAY")
        self.status_var.set("❌ Analysis failed")
        
        msg = f"Analysis failed!\n\n❌ Error: {error}" if error else "Analysis failed!\n\nCheck the output above for details."
        messagebox.showerror("Error", msg)
    
    def open_folder(self, path):
        """Open folder in file explorer"""
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        
        try:
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
            self.status_var.set(f"Opened folder: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{e}")
    
    def view_report(self):
        """View latest diagnosis report"""
        try:
            reports = list(Path("outputs/diagnosis").glob("*_report.txt"))
            if reports:
                latest = max(reports, key=lambda p: p.stat().st_mtime)
                if sys.platform == 'win32':
                    os.startfile(latest)
                else:
                    subprocess.Popen(['open', latest] if sys.platform == 'darwin' else ['xdg-open', latest])
                self.status_var.set(f"Opened: {latest.name}")
            else:
                messagebox.showinfo(
                    "No Reports Found",
                    "No diagnosis reports found yet.\n\nRun an analysis first to generate a report."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Could not open report:\n{e}")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🏥 SCOLIOSIS AI - HELP

📋 HOW TO USE:
1. Click 'Browse' to select an X-ray image
2. (Optional) Adjust confidence threshold (0.25 recommended)
3. Click 'ANALYZE X-RAY' to start diagnosis
4. Wait for analysis to complete
5. View results in the output log or click 'View Latest Report'

📊 WHAT YOU GET:
• Spine detection with bounding boxes
• Cobb angle measurement
• Severity classification (Normal/Mild/Moderate/Severe)
• Annotated image with visual markers
• Detailed text report
• JSON summary for integration

📁 OUTPUT LOCATION:
All results are saved in: outputs/diagnosis/

🎯 CONFIDENCE THRESHOLD:
• 0.25 = Recommended (balanced)
• Lower = More detections (less strict)
• Higher = Fewer detections (more strict)

💡 TIPS:
• Use clear, high-quality X-ray images
• Ensure spine is visible in the image
• Results include annotated images for visual verification

ℹ️ For more information, see README.md
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Scoliosis AI")
        help_window.geometry("600x500")
        
        help_text_widget = scrolledtext.ScrolledText(
            help_window,
            font=("Arial", 10),
            wrap=tk.WORD,
            padx=20,
            pady=20
        )
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        help_text_widget.insert(tk.END, help_text.strip())
        help_text_widget.config(state='disabled')
        
        close_btn = tk.Button(
            help_window,
            text="Close",
            command=help_window.destroy,
            bg=ModernStyle.PRIMARY,
            fg=ModernStyle.WHITE,
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        close_btn.pack(pady=10)


def main():
    """Launch the simplified application"""
    root = tk.Tk()
    app = ScoliosisAISimple(root)
    root.mainloop()


if __name__ == "__main__":
    main()
