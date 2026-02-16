"""
Scoliosis AI - User-Friendly GUI Launcher
Simple graphical interface for diagnosis, training, and analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
from pathlib import Path
import sys
import os


class ScoliosisAILauncher:
    """Main GUI application for Scoliosis AI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Scoliosis AI - Professional Diagnosis System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Set icon if exists
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.selected_image = tk.StringVar()
        self.model_choice = tk.StringVar(value="all")
        self.confidence = tk.DoubleVar(value=0.25)
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the user interface"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🏥 Scoliosis AI Diagnosis System",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main content area
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook (tabs)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Diagnosis
        diagnosis_tab = ttk.Frame(notebook, padding="20")
        notebook.add(diagnosis_tab, text="📊 Diagnosis")
        self.create_diagnosis_tab(diagnosis_tab)
        
        # Tab 2: Training
        training_tab = ttk.Frame(notebook, padding="20")
        notebook.add(training_tab, text="🎓 Training")
        self.create_training_tab(training_tab)
        
        # Tab 3: Evaluation
        eval_tab = ttk.Frame(notebook, padding="20")
        notebook.add(eval_tab, text="📈 Evaluation")
        self.create_evaluation_tab(eval_tab)
        
        # Tab 4: Tools
        tools_tab = ttk.Frame(notebook, padding="20")
        notebook.add(tools_tab, text="🔧 Tools")
        self.create_tools_tab(tools_tab)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_diagnosis_tab(self, parent):
        """Create diagnosis interface"""
        
        # Instructions
        instructions = tk.Label(
            parent,
            text="Upload an X-ray image to get instant scoliosis diagnosis with detailed report",
            font=("Arial", 11),
            fg="#555"
        )
        instructions.pack(pady=(0, 20))
        
        # Image selection
        image_frame = ttk.LabelFrame(parent, text="1. Select X-ray Image", padding="15")
        image_frame.pack(fill=tk.X, pady=10)
        
        image_entry = ttk.Entry(image_frame, textvariable=self.selected_image, width=60)
        image_entry.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        browse_btn = ttk.Button(
            image_frame,
            text="Browse...",
            command=self.browse_image
        )
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Model settings
        settings_frame = ttk.LabelFrame(parent, text="2. Analysis Settings", padding="15")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Model choice
        tk.Label(settings_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.model_choice,
            values=["yolo", "all"],
            state="readonly",
            width=20
        )
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=10)
        
        # Confidence threshold
        tk.Label(settings_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, pady=5)
        confidence_scale = ttk.Scale(
            settings_frame,
            from_=0.1,
            to=0.9,
            variable=self.confidence,
            orient=tk.HORIZONTAL,
            length=200
        )
        confidence_scale.grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
        
        confidence_label = tk.Label(
            settings_frame,
            textvariable=self.confidence,
            width=5
        )
        confidence_label.grid(row=1, column=2, pady=5)
        
        # Run button
        run_frame = ttk.Frame(parent)
        run_frame.pack(pady=20)
        
        self.run_diagnosis_btn = ttk.Button(
            run_frame,
            text="🔬 Run Diagnosis",
            command=self.run_diagnosis,
            style="Accent.TButton",
            width=30
        )
        self.run_diagnosis_btn.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(
            parent,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=10)
        
        # Output log
        log_frame = ttk.LabelFrame(parent, text="Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.diagnosis_log = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Consolas", 9)
        )
        self.diagnosis_log.pack(fill=tk.BOTH, expand=True)
        
        # Result buttons
        result_frame = ttk.Frame(parent)
        result_frame.pack(pady=10)
        
        ttk.Button(
            result_frame,
            text="📂 Open Results Folder",
            command=lambda: self.open_folder("outputs/diagnosis")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            result_frame,
            text="📄 View Report",
            command=self.view_report
        ).pack(side=tk.LEFT, padx=5)
    
    def create_training_tab(self, parent):
        """Create training interface"""
        
        info = tk.Label(
            parent,
            text="Train custom scoliosis detection models on your dataset",
            font=("Arial", 11),
            fg="#555"
        )
        info.pack(pady=(0, 20))
        
        # Dataset info
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Configuration", padding="15")
        dataset_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            dataset_frame,
            text="📁 Validate Dataset",
            command=self.validate_dataset
        ).pack(pady=5)
        
        self.dataset_info = tk.Label(
            dataset_frame,
            text="Click validate to check dataset",
            font=("Arial", 9),
            justify=tk.LEFT
        )
        self.dataset_info.pack(pady=10)
        
        # Training options
        options_frame = ttk.LabelFrame(parent, text="Training Options", padding="15")
        options_frame.pack(fill=tk.X, pady=10)
        
        self.epochs_var = tk.IntVar(value=30)
        self.batch_var = tk.IntVar(value=16)
        self.imgsz_var = tk.IntVar(value=640)
        
        tk.Label(options_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(options_frame, from_=10, to=200, textvariable=self.epochs_var, width=15).grid(row=0, column=1, pady=5, padx=10)
        
        tk.Label(options_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(options_frame, from_=4, to=64, textvariable=self.batch_var, width=15).grid(row=1, column=1, pady=5, padx=10)
        
        tk.Label(options_frame, text="Image Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(options_frame, textvariable=self.imgsz_var, values=[320, 640, 1024], state="readonly", width=13).grid(row=2, column=1, pady=5, padx=10)
        
        # Training buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=20)
        
        ttk.Button(
            btn_frame,
            text="⚡ Quick Training (Fast)",
            command=self.start_training_fast,
            width=30
        ).pack(pady=5)
        
        ttk.Button(
            btn_frame,
            text="🎯 High Accuracy Training (Recommended)",
            command=self.start_training_high_accuracy,
            width=30
        ).pack(pady=5)
        
        ttk.Button(
            btn_frame,
            text="📊 Monitor Training",
            command=self.monitor_training,
            width=30
        ).pack(pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(parent, text="Training Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Consolas", 9)
        )
        self.training_log.pack(fill=tk.BOTH, expand=True)
    
    def create_evaluation_tab(self, parent):
        """Create evaluation interface"""
        
        info = tk.Label(
            parent,
            text="Evaluate model performance with statistical metrics",
            font=("Arial", 11),
            fg="#555"
        )
        info.pack(pady=(0, 20))
        
        # Quick evaluation
        quick_frame = ttk.LabelFrame(parent, text="Quick Evaluation", padding="15")
        quick_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            quick_frame,
            text="🎯 Run Model Validation",
            command=self.run_validation,
            width=30
        ).pack(pady=5)
        
        ttk.Button(
            quick_frame,
            text="📈 Generate Evaluation Report",
            command=self.generate_eval_report,
            width=30
        ).pack(pady=5)
        
        # Advanced evaluation
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Analysis", padding="15")
        advanced_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            advanced_frame,
            text="📊 ROC Curves & Confusion Matrix",
            command=self.generate_roc,
            width=30
        ).pack(pady=5)
        
        ttk.Button(
            advanced_frame,
            text="📉 Bland-Altman Analysis",
            command=self.bland_altman,
            width=30
        ).pack(pady=5)
        
        # Results
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.eval_log = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            font=("Consolas", 9)
        )
        self.eval_log.pack(fill=tk.BOTH, expand=True)
    
    def create_tools_tab(self, parent):
        """Create tools interface"""
        
        # Project management
        project_frame = ttk.LabelFrame(parent, text="Project Management", padding="15")
        project_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            project_frame,
            text="✨ Setup & Validate Project",
            command=self.run_setup,
            width=35
        ).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(
            project_frame,
            text="🧹 Cleanup Temporary Files",
            command=self.cleanup_project,
            width=35
        ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(
            project_frame,
            text="📦 Export Model Package",
            command=self.export_model,
            width=35
        ).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(
            project_frame,
            text="📋 View System Info",
            command=self.show_system_info,
            width=35
        ).grid(row=1, column=1, padx=5, pady=5)
        
        # Documentation
        docs_frame = ttk.LabelFrame(parent, text="Documentation", padding="15")
        docs_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            docs_frame,
            text="📖 Quick Start Guide",
            command=lambda: self.open_doc("QUICKSTART.md"),
            width=35
        ).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(
            docs_frame,
            text="🎓 PhD Research Roadmap",
            command=lambda: self.open_doc("PHD_ROADMAP.md"),
            width=35
        ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(
            docs_frame,
            text="💡 Commands Reference",
            command=lambda: self.open_doc("COMMANDS.txt"),
            width=35
        ).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(
            docs_frame,
            text="📊 Project Summary",
            command=lambda: self.open_doc("PROJECT_SUMMARY.md"),
            width=35
        ).grid(row=1, column=1, padx=5, pady=5)
        
        # About
        about_frame = ttk.LabelFrame(parent, text="About", padding="15")
        about_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        about_text = """
        🏥 Scoliosis AI Detection System
        
        Version: 2.0 (PhD Edition)
        
        Features:
        • YOLOv8-based spine detection
        • Geometric Cobb angle measurement
        • Multi-task deep learning architecture
        • Clinical-grade evaluation metrics
        • Publication-ready visualizations
        
        Built with: PyTorch, Ultralytics, OpenCV, NumPy
        
        For research and clinical use
        """
        
        about_label = tk.Label(
            about_frame,
            text=about_text,
            justify=tk.LEFT,
            font=("Arial", 10)
        )
        about_label.pack(pady=10)
    
    # === Command Methods ===
    
    def browse_image(self):
        """Browse for X-ray image"""
        filename = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.selected_image.set(filename)
    
    def run_diagnosis(self):
        """Run diagnosis on selected image"""
        if not self.selected_image.get():
            messagebox.showwarning("No Image", "Please select an X-ray image first!")
            return
        
        if not Path(self.selected_image.get()).exists():
            messagebox.showerror("File Not Found", f"Image not found: {self.selected_image.get()}")
            return
        
        # Disable button and start progress
        self.run_diagnosis_btn.config(state='disabled')
        self.progress.start()
        self.status_var.set("Running diagnosis...")
        
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
                "--model", self.model_choice.get(),
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
                self.root.after(0, self._update_diagnosis_log, line)
            
            process.wait()
            
            if process.returncode == 0:
                self.root.after(0, self._diagnosis_complete)
            else:
                self.root.after(0, self._diagnosis_error)
                
        except Exception as e:
            self.root.after(0, self._diagnosis_error, str(e))
    
    def _update_diagnosis_log(self, text):
        """Update diagnosis log"""
        self.diagnosis_log.insert(tk.END, text)
        self.diagnosis_log.see(tk.END)
    
    def _diagnosis_complete(self):
        """Handle diagnosis completion"""
        self.progress.stop()
        self.run_diagnosis_btn.config(state='normal')
        self.status_var.set("Diagnosis complete!")
        messagebox.showinfo(
            "Success",
            "Diagnosis completed successfully!\n\nResults saved to: outputs/diagnosis/"
        )
    
    def _diagnosis_error(self, error=None):
        """Handle diagnosis error"""
        self.progress.stop()
        self.run_diagnosis_btn.config(state='normal')
        self.status_var.set("Error occurred")
        msg = f"Diagnosis failed!\n\nError: {error}" if error else "Diagnosis failed!"
        messagebox.showerror("Error", msg)
    
    def open_folder(self, path):
        """Open folder in file explorer"""
        folder = Path(path)
        if folder.exists():
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
        else:
            messagebox.showwarning("Folder Not Found", f"Folder does not exist: {path}")
    
    def view_report(self):
        """View latest diagnosis report"""
        reports = list(Path("outputs/diagnosis").glob("*_report.txt"))
        if reports:
            latest = max(reports, key=lambda p: p.stat().st_mtime)
            if sys.platform == 'win32':
                os.startfile(latest)
            else:
                subprocess.Popen(['open', latest] if sys.platform == 'darwin' else ['xdg-open', latest])
        else:
            messagebox.showinfo("No Reports", "No diagnosis reports found. Run a diagnosis first!")
    
    def validate_dataset(self):
        """Validate dataset configuration"""
        self.dataset_info.config(text="Validating dataset...")
        # Implementation here
        self.dataset_info.config(text="Dataset validation not yet implemented")
    
    def start_training_fast(self):
        """Start fast training (30 epochs, 320px, nano model)"""
        response = messagebox.askyesno(
            "Quick Training",
            "This will train a FAST model:\n\n"
            "• Model: YOLOv8-Nano (smallest)\n"
            "• Epochs: ~30\n"
            "• Image Size: 320px\n"
            "• Time: ~1-2 hours\n"
            "• Accuracy: Good\n\n"
            "Start training?"
        )
        if response:
            try:
                self.training_log.insert(tk.END, "Starting fast training...\n")
                subprocess.Popen(
                    [sys.executable, "src/train_yolo.py"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )
                messagebox.showinfo("Training Started", "Training started in new window.\nCheck the console for progress.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def start_training_high_accuracy(self):
        """Start high accuracy training (200 epochs, 1024px, medium model)"""
        response = messagebox.askyesno(
            "High Accuracy Training",
            "This will train a HIGH ACCURACY model:\n\n"
            "• Model: YOLOv8-Medium (25.9M params)\n"
            "• Epochs: ~200\n"
            "• Image Size: 1024px\n"
            "• Augmentation: Advanced (Mosaic, MixUp, RandAugment)\n"
            "• Dataset: 100% (full dataset)\n"
            "• Time: ~8-12 hours (CPU) or ~2-3 hours (GPU)\n"
            "• Accuracy: MAXIMUM\n\n"
            "⚠️ This will take significant time. Continue?"
        )
        if response:
            try:
                self.training_log.insert(tk.END, "Starting high accuracy training...\n")
                self.training_log.insert(tk.END, "Configuration: YOLOv8-Medium, 200 epochs, 1024px\n")
                self.training_log.insert(tk.END, "This may take several hours...\n\n")
                subprocess.Popen(
                    [sys.executable, "train_advanced.py", "--mode", "train"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )
                messagebox.showinfo(
                    "Training Started",
                    "High accuracy training started in new window.\n\n"
                    "Check the console for real-time progress.\n"
                    "Model will be saved to:\n"
                    "models/detection/scoliosis_yolo_high_accuracy/weights/best.pt"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def start_training(self):
        """Legacy training function"""
        self.start_training_fast()
    
    def monitor_training(self):
        """Open training monitor"""
        folder = Path("models/detection")
        if folder.exists():
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
        else:
            messagebox.showinfo("Not Found", "No training results found yet.")
    
    def run_validation(self):
        """Run model validation"""
        self.eval_log.insert(tk.END, "Running validation...\n")
    
    def generate_eval_report(self):
        """Generate evaluation report"""
        self.eval_log.insert(tk.END, "Generating evaluation report...\n")
    
    def generate_roc(self):
        """Generate ROC curves"""
        self.eval_log.insert(tk.END, "Generating ROC curves...\n")
    
    def bland_altman(self):
        """Generate Bland-Altman plot"""
        self.eval_log.insert(tk.END, "Generating Bland-Altman analysis...\n")
    
    def run_setup(self):
        """Run setup and validation"""
        if sys.platform == 'win32':
            subprocess.Popen(["SETUP_VALIDATE.bat"], shell=True)
        else:
            messagebox.showinfo("Info", "Please run: ./SETUP_VALIDATE.bat")
    
    def cleanup_project(self):
        """Cleanup temporary files"""
        result = messagebox.askyesno(
            "Confirm Cleanup",
            "This will remove temporary files, caches, and test outputs.\n\nContinue?"
        )
        if result:
            if sys.platform == 'win32':
                subprocess.Popen(["CLEANUP.bat"], shell=True)
            else:
                messagebox.showinfo("Info", "Please run: ./CLEANUP.bat")
    
    def export_model(self):
        """Export model package"""
        messagebox.showinfo("Export", "Model export feature coming soon!")
    
    def show_system_info(self):
        """Show system information"""
        try:
            import torch
            info = f"""
System Information:

Python: {sys.version.split()[0]}
PyTorch: {torch.__version__}
CUDA Available: {torch.cuda.is_available()}
Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}

Project Path: {Path('.').absolute()}
            """
            messagebox.showinfo("System Info", info.strip())
        except Exception as e:
            messagebox.showerror("Error", f"Could not get system info:\n{e}")
    
    def open_doc(self, filename):
        """Open documentation file"""
        doc_path = Path(filename)
        if doc_path.exists():
            if sys.platform == 'win32':
                os.startfile(doc_path)
            else:
                subprocess.Popen(['open', doc_path] if sys.platform == 'darwin' else ['xdg-open', doc_path])
        else:
            messagebox.showwarning("File Not Found", f"Documentation file not found: {filename}")


def main():
    """Launch the application"""
    root = tk.Tk()
    app = ScoliosisAILauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
