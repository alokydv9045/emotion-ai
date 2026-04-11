# 🚀 Deployment Guide: AI Emotion Detection System

This guide provides instructions on how to deploy the AI Emotion Detection system to various platforms.

## 📋 Prerequisites

- **Python 3.9+**
- **Git**
- **FER2013 Model File**: Ensure `models/emotion_model.h5` is present.

---

## ☁️ Option 1: Streamlit Community Cloud (Recommended)

1. **Push to GitHub**:
   - Create a new repository on GitHub.
   - Commit and push all files (including `models/` and `dataset/` if you want them hosted).
   - *Note: For the 70MB model, ensuring it's in the repo is fine, but for larger files (>100MB) use Git LFS.*

2. **Deploy on Streamlit**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Connect your GitHub account.
   - Select your repository, branch (`main`), and main file path (`app.py`).
   - Click **Deploy**.

---

## 🐳 Option 2: Docker Deployment

1. **Create a `Dockerfile`**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**:
   ```bash
   docker build -t emotion-ai .
   docker run -p 8501:8501 emotion-ai
   ```

---

## 🔧 Option 3: Manual Server Deployment (Ubuntu/Linux)

1. **Clone and Setup**:
   ```bash
   git clone <your-repo-url>
   cd emotion-ai
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install System Dependencies (OpenCV)**:
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```

3. **Run using a Process Manager (PM2)**:
   ```bash
   pm2 start "streamlit run app.py" --name emotion-ai
   ```

---

## 🛠️ Optimizations for Production

- **Voice Support**: `pyttsx3` requires a system display and audio drivers. For server-side cloud deployments (like Streamlit Cloud), the voice feedback may not work unless using a browser-based TTS (Web Speech API).
- **Gather Usage Stats**: Always run with `--browser.gatherUsageStats false` in production to avoid prompts.
- **Resources**: The CNN model requires ~500MB of RAM. Ensure your VPS or Cloud instance has at least 1GB of memory.
