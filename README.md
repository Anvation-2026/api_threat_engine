# 🛡️ CY-02 Sentinel: Decoupled API Threat Intelligence Engine

**CY-02 Sentinel** is a 3-tier, real-time API Security Operations Center (SOC) and behavioral threat detection system built by **Team BYTEFORCE**.

Designed as a modern, adaptive alternative to static Web Application Firewalls (WAFs), CY-02 combines heuristic rule engines, unsupervised Machine Learning (**Isolation Forest**), and **Markov route sequence models** to detect and block automated attack campaigns (such as credential stuffing, scrapers, and enumeration) in real time without impacting legitimate enterprise traffic.

---

## 🌟 Key Features & Capabilities

* **3-Tier Decoupled Architecture:** Completely separates the target e-commerce application, log ingestion API, and Streamlit SOC dashboard for zero-latency backend impact.
* **Multi-Model Composite Threat Scoring Pipeline:**
  $$\text{Composite Risk} = (0.40 \times \text{Rules}) + (0.35 \times \text{IsoForest ML}) + (0.25 \times \text{Markov Sequence})$$
* **Burp Suite Attack Vector Detection:** Detects credential stuffing, web scrapers, resource enumeration, and multi-IP low-and-slow attacks with simulated offensive triggers.
* **Explainable Threat Forensics:** Plain-language evidence cards detailing login pressure, HTTP failure rates ($401$/$404$), timing regularity ($CV$), and route traversal anomalies.
* **Zero False Positive Shield:** Per-client baseline Z-score modeling ensures heavy enterprise B2B partner traffic (e.g., `partner_acme`) is never falsely blocked.

---

## 🏗️ System Architecture & Services

The system is split into **3 decoupled services** running on independent ports:

1. **Target App Simulator (`quick_store.py` - Port 8000):** E-commerce frontend with built-in attack vector simulators that fire live Burp Suite-style traffic bursts.
2. **Log Ingestion Service (`log_receiver.py` - Port 9000):** Asynchronous FastAPI service that parses HTTP telemetry and appends labeled records to `data/live_logs.csv`.
3. **Security Operations Center (`app.py` - Port 8501):** Streamlit dashboard executing real-time feature extraction, risk matrices, and live metric updates.

---

## 📂 Complete Directory Structure

```text
api-threat-detection/
├── data/
│   ├── history_logs.csv        # Historical baseline memory for ML fitting
│   └── live_logs.csv           # Real-time ingested telemetry stream
├── app.py                       # Streamlit Security SOC Dashboard (Port 8501)
├── log_receiver.py              # FastAPI Log Ingestion API (Port 9000)
├── quick_store.py               # Target Store App & Attack Simulator (Port 8000)
├── feature_extraction.py        # 1-Minute Window Aggregation Pipeline
├── rule_detectors.py            # Heuristic Rule Engine
├── baseline_models.py           # Adaptive Per-Client Baseline Z-Score Engine
├── sequence_model.py            # Markov Route Sequence Detector
├── isolation_forest.py          # Isolation Forest ML Threat Scoring Engine
├── sentinel_sdk.py              # Plug-and-Play Middleware Integration SDK
├── requirements.txt             # Python Package Dependencies
└── README.md                    # Project Documentation


##⚡ Quick Start Guide
1. Installation & Environment Setup

Clone the repository and install all required dependencies:

git clone [https://github.com/Sahajalkatiyar/Anvation-project.git](https://github.com/Sahajalkatiyar/Anvation-project.git)
cd api-threat-detection
pip install -r requirements.txt

2. Launching the System

Open 3 separate terminal windows and execute the commands below in order:

    Terminal 1 (Log Ingestion API): python log_receiver.py
    Terminal 2 (Streamlit Security SOC):python -m streamlit run app.py
    Terminal 3 (Target App Simulator):python quick_store.py

##🧪 How to Demo Live Attack Vectors

    Open the Target Application in your browser: http://127.0.0.1:8000.
    Click any attack simulator button to fire offensive traffic bursts:
        1. Credential Stuffing (Critical): Fires 20 rapid failed login calls returning HTTP 401.
        2. Low & Slow Attack: Fires subtle login attempts across distributed client IDs.
        3. Product Scraper: Sweeps 25 product catalog pages sequentially (/api/products?page=N).
        4. User ID Enumeration: Scans user profile paths returning HTTP 404 errors.
    Open the CY-02 SOC Dashboard: http://localhost:8501.

    Click the 🔄 Refresh Data button in the sidebar to observe:
       1. Instant increase in CRITICAL THREATS (>70) metric.
       2. Red high-risk bubble clusters in the Interactive Behavioral Threat Matrix.
       3. Updated forensic evidence cards detailing the exact rule/ML triggers.

##🤖 Why Machine Learning Is Used
  While traditional WAFs rely solely on static rate limits, CY-02 Sentinel utilizes an Isolation Forest ML Model:
      1.Unsupervised Anomaly Detection: Learns normal traffic behavior without requiring human-labeled training data.

##🛠️ Tech Stack & Team
  1.Dashboard & Visualizations: Streamlit, Plotly, HTML5/CSS3
  2.Backend Services: FastAPI, Uvicorn, HTTPX
  3.Machine Learning & Analytics: Scikit-Learn (Isolation Forest), Pandas, NumPy

##Project Team: Team BYTEFORCE
  
