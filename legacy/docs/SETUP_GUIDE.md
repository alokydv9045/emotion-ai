# 🛠️ Emotion-AI: Setup & Environment Guide

If your IDE is showing "Module Not Found" errors or warning markers, follow these steps to synchronize your environment.

## 1. Install Dependencies
Open your terminal in the project root and run:
```bash
pip install -r requirements.txt
```

## 2. Requirements Breakdown
The system relies on several key libraries:
- **`streamlit`**: The web engine.
- **`opencv-python`**: The vision engine.
- **`tensorflow`**: The AI engine.
- **`gtts` / `pyttsx3`**: The voice engine.
- **`plotly`**: The analytics engine.

## 3. IDE Synchronization (VS Code / PyCharm)
If your terminal shows a successful install but your IDE still shows "Module Not Found" warnings:

1.  **Open Command Palette** (`Ctrl+Shift+P` on Windows).
2.  **Select Interpreter**: Search for "Python: Select Interpreter".
3.  **Choose the correct environment**: Pick the one where you ran `pip install`.
4.  **Restart Language Server**: If warnings persist, restart VS Code.

> [!IMPORTANT]
> **Python 3.13 Warning**: Some libraries like `tensorflow` might have compatibility lag on Python 3.13. If `pip install` fails specifically for tensorflow, consider using Python 3.10 or 3.12 for maximum stability.

## 4. Troubleshooting
- **Missing `cv2`**: This is provided by `opencv-python`.
- **Missing `PIL`**: This is provided by `Pillow`.
- **Audio Errors**: If the voice fails, ensure `pygame` or `playsound` is installed.

## 5. Run the Project
Once dependencies are synced, start the engine:
```bash
streamlit run app.py
```
