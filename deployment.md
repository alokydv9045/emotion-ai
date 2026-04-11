# 🚀 Deployment Guide: Emotion Detection System

This guide walks you through the process of deploying the **emotion detection system** across different environments, from local workstations to cloud-scale platforms like Render and Vercel.

---

## 📋 Prerequisites

Before you begin, ensure you have the following ready:
- **GitHub Repository**: The code must be pushed to a public or private GitHub repository.
- **API Keys**: An active **ElevenLabs API Key** for voice features.
- **Accounts**: Registered accounts on [Render.com](https://render.com) and [Vercel.com](https://vercel.com).

---

## 🏠 Option 1: Local Workstation Deployment

Perfect for development, testing, and high-performance sessions using local GPU acceleration.

### 🛠️ Step 1: Environment Calibration
1.  **Clone the Repository**:
    ```powershell
    git clone https://github.com/alokyadav9045/Emotion-ai.git
    cd Emotion-ai
    ```
2.  **Initialize Virtual Environment**:
    ```powershell
    python -m venv .venv
    ```
3.  **Install Neural Dependencies**:
    ```powershell
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

### 🎨 Step 2: Frontend Asset Compilation
1.  **Navigate and Build**:
    ```powershell
    cd frontend
    npm install
    npm run build
    cd ..
    ```

### 🚀 Step 3: Production Launch
1.  **Configure Secrets**: Create a `.env` file and add `ELEVENLABS_API_KEY=your_key`.
2.  **Run Launcher**:
    ```powershell
    ./launch_production.bat
    ```

---

## ☁️ Option 2: Cloud Deployment (Render + Vercel)

Deploy your emotion detection system online for global access. This setup uses a **Decoupled Architecture**: the Backend stays on Render, and the Frontend stays on Vercel.

### 🚄 A. Backend Deployment (Render)

1.  **Create a New Web Service**:
    - Log in to [Render Dashboard](https://dashboard.render.com/).
    - Click **New +** > **Web Service**.
    - Connect your GitHub repository.
2.  **Service Configuration**:
    - **Name**: `emotion-ai-backend`
    - **Environment**: `Python 3`
    - **Region**: Choose the one closest to you.
    - **Branch**: `main`
3.  **Build & Start Commands**:
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4.  **Environment Variables**:
    - Add `ELEVENLABS_API_KEY`: `your_actual_api_key_here`
    - Add `PYTHON_VERSION`: `3.10.12` (or your preferred version)
5.  **Deploy**: Click **Create Web Service**. 
    - *Note the "Onrender" URL provided (e.g., `https://emotion-ai-backend.onrender.com`). You will need this for the frontend.*

### ⚡ B. Frontend Deployment (Vercel)

1.  **Import Project**:
    - Log in to [Vercel Dashboard](https://vercel.com/dashboard).
    - Click **Add New** > **Project**.
    - Import your GitHub repository.
2.  **Project Settings**:
    - **Framework Preset**: `Vite`
    - **Root Directory**: `frontend` (⚠️ **CRITICAL**: Set this to the `frontend` folder).
3.  **Environment Variables**:
    - Under the **Environment Variables** section, add:
    - **Key**: `VITE_API_URL`
    - **Value**: `https://your-render-app-url.onrender.com` (The URL from Step A.5)
4.  **Deploy**: Click **Deploy**.

---

## 🔄 Step 4: Verification (The Live Link)

Once both services are "Live":
1.  Visit your Vercel URL (e.g., `https://emotion-ai-frontend.vercel.app`).
2.  Check the **HUD Status**: It should show `LINK ACTIVE`.
3.  Try a **Live Scan**: If the webcam stream works and scores appear, the Vercel-to-Render bridge is successful.

---

## ⚠️ Cloud-Specific Gotchas

### 1. Cold Starts (Render Free Tier)
Render's free tier spins down services after inactivity. If you open the site and it stays on "WAITING FOR NEURAL SOURCE" for a minute, the backend is likely waking up. 

### 2. CORS Issues
If you see "LINK LOST" or "Connection Refused" in the console, ensure the `backend/main.py` CORS middleware is allowing the Vercel domain. The current project is configured with `allow_origins=["*"]`, which is compatible with all cloud deployments.

### 3. Missing Models
Ensure your `.h5` model files (in `backend/models/`) are committed and pushed to GitHub. Render needs these files to initialize the neural engine.

---
**System Document v6.0 | Cloud-Ready Revision**
