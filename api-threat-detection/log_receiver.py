import csv
import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="CY-02 Sentinel Log Ingestion API")

os.makedirs("data", exist_ok=True)
LOG_FILE = os.path.join("data", "live_logs.csv")

class LogPayload(BaseModel):
    timestamp: str
    client_id: str
    ip: str
    endpoint: str
    status: int
    user_agent: str
    username: str = "guest"

def write_to_csv(log: LogPayload):
    ua_lower = log.user_agent.lower()
    cid_lower = log.client_id.lower()
    
    # 🎯 Enhanced Label Classifier
    label = "normal"
    if "stuffer" in cid_lower or "credential" in ua_lower:
        label = "credential_stuffing"
    elif "stealth" in cid_lower or "stealth" in ua_lower:
        label = "low_and_slow"
    elif "scraper" in cid_lower or "scraper" in ua_lower:
        label = "scraper"
    elif "enum" in cid_lower or "enum" in ua_lower:
        label = "enumeration"
    elif "bot" in ua_lower or "python" in ua_lower:
        label = "credential_stuffing" if log.endpoint == "/api/login" else "scraper"

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "client_id", "ip", "endpoint", "status", "user_agent", "username", "label"])
        writer.writerow([
            log.timestamp,
            log.client_id,
            log.ip,
            log.endpoint,
            log.status,
            log.user_agent,
            log.username,
            label
        ])

@app.post("/api/v1/logs")
async def receive_log(log: LogPayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_to_csv, log)
    return {"status": "INGESTED"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000, reload=False)