# 🏥 Scoliosis AI - Which Version to Use?

## ✅ **SIMPLIFIED VERSION (Recommended for Most Users)**

**File**: `launcher_simple.py`  
**Launch**: Double-click `LAUNCH.bat`

### What it does:
- ✅ Upload X-ray images
- ✅ AI-powered diagnosis
- ✅ Cobb angle measurement  
- ✅ Severity classification
- ✅ Visual reports
- ✅ Clean, simple interface

### Perfect for:
- Clinical use
- Quick diagnosis
- Testing X-rays
- Daily operations
- Non-technical users

### Screenshots:
- Single-page interface
- Just 3 steps: Select → Analyze → View Results
- No complex features

---

## 🔬 **FULL VERSION (For Researchers & Developers)**

**File**: `launcher.py` or `launcher_enhanced.py`  
**Launch**: Run `venv\Scripts\python.exe launcher.py`

### Additional features:
- 📊 Training custom models
- 📈 Model evaluation & validation
- 🔬 Data Science analysis
- 📉 Statistical visualizations
- 📋 ROC curves, Bland-Altman plots
- 🎓 PhD-level metrics

### Perfect for:
- Research projects
- Model development
- Statistical validation
- Academic papers
- Advanced analysis

---

## 🚀 **Quick Start**

### For Simple Diagnosis (Most Users):
```
1. Double-click: LAUNCH.bat
2. GUI opens automatically
3. Select X-ray image
4. Click "ANALYZE X-RAY"
5. Done!
```

### For Advanced Features:
```
1. Open terminal
2. cd scoliosis_ai
3. venv\Scripts\python.exe launcher.py
4. Use all 5 tabs
```

---

## 📊 **Comparison**

| Feature | Simple | Full |
|---------|--------|------|
| X-ray Diagnosis | ✅ | ✅ |
| Cobb Angle | ✅ | ✅ |
| Severity Classification | ✅ | ✅ |
| Visual Reports | ✅ | ✅ |
| Model Training | ❌ | ✅ |
| Model Evaluation | ❌ | ✅ |
| Statistical Analysis | ❌ | ✅ |
| Data Science Tools | ❌ | ✅ |
| ROC Curves | ❌ | ✅ |
| Bland-Altman | ❌ | ✅ |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Speed** | Fast | Medium |
| **Interface** | Clean | Complex |

---

## 💡 **Recommendation**

### Use **SIMPLIFIED** if you:
- ✅ Just need X-ray diagnosis
- ✅ Want quick results
- ✅ Prefer simple interface
- ✅ Don't need training/research features
- ✅ Are a clinical user

### Use **FULL** version if you:
- 🔬 Need to train custom models
- 📊 Want statistical analysis
- 📈 Need publication-ready graphs
- 🎓 Are doing research
- 💻 Are a developer/data scientist

---

## 🔄 **Switching Versions**

### To use Simple (default):
```batch
# LAUNCH.bat already configured
Double-click LAUNCH.bat
```

### To use Full:
```batch
cd scoliosis_ai
venv\Scripts\python.exe launcher.py
```

### To change default:
Edit `LAUNCH.bat` and change:
```batch
# Simple (default):
venv\Scripts\python.exe launcher_simple.py

# Full:
venv\Scripts\python.exe launcher.py
```

---

## 📁 **File Reference**

```
scoliosis_ai/
├── LAUNCH.bat                    # Quick launcher (uses simple)
├── launcher_simple.py            # SIMPLIFIED VERSION ⭐
├── launcher.py                   # FULL VERSION (enhanced)
├── launcher_enhanced.py          # FULL VERSION (backup)
├── launcher_old_backup.py        # Original (backup)
└── diagnose.py                   # Core diagnosis engine (both use this)
```

---

## ❓ **FAQ**

**Q: Which version should I use?**  
A: If you're just diagnosing X-rays → Use SIMPLIFIED (default)

**Q: Can I switch between versions?**  
A: Yes! Both are included. Just run the launcher you prefer.

**Q: Will simple version get all features eventually?**  
A: No - it's intentionally simple. Use full version if you need advanced features.

**Q: Do both versions give same diagnosis results?**  
A: Yes! Both use the same AI engine (diagnose.py). Only the interface differs.

**Q: Which is faster?**  
A: Simple version - fewer tabs means faster loading and cleaner UI.

---

## 🎯 **Current Default**

✅ **SIMPLIFIED VERSION** is now the default  
📁 `LAUNCH.bat` → `launcher_simple.py`

This was changed because:
- Most users only need diagnosis
- Simpler = fewer errors
- Faster to load
- Easier to use

**Full version is still available** - just run `launcher.py` directly!

---

**Need help?** Check [README.md](README.md) or [FEATURES.md](FEATURES.md)
