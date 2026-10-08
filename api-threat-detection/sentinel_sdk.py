"""
CY-02 Sentinel: Plug-and-Play Middleware SDK
Allows developers to attach CY-02 threat detection to any FastAPI / Python Web App in 2 lines.
"""

import time
import pandas as pd
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class CY02SentinelMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, threat_engine=None, risk_threshold=70.0):
        super().__init__(app)
        self.risk_threshold = risk_threshold
        self.engine = threat_engine

    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        # 1. Extract request metadata
        client_ip = request.client.host if request.client else "127.0.0.1"
        client_id = request.headers.get("X-Client-ID", f"client_{client_ip}")
        endpoint = request.url.path
        user_agent = request.headers.get("User-Agent", "Unknown")

        # 2. Simulated In-Memory Quick Risk Evaluation (<15ms latency)
        # In production, this checks Redis window counters
        is_suspicious_bot = "bot" in user_agent.lower() or "python" in user_agent.lower()
        is_sensitive_endpoint = endpoint in ["/api/login", "/api/checkout", "/api/admin/export"]
        
        # Calculate instant request score
        req_risk_score = 0.0
        if is_suspicious_bot:
            req_risk_score += 40.0
        if is_sensitive_endpoint:
            req_risk_score += 35.0
            
        latency_ms = (time.time() - start_time) * 1000

        # 3. Active Enforcement Block
        if req_risk_score >= self.risk_threshold:
            return JSONResponse(
                status_code=429,
                content={
                    "status": "BLOCKED_BY_CY02_SENTINEL",
                    "client_id": client_id,
                    "ip": client_ip,
                    "risk_score": req_risk_score,
                    "reason": f"High risk pattern detected on sensitive endpoint '{endpoint}'",
                    "latency_ms": round(latency_ms, 2)
                }
            )

        # 4. Request Allowed: Pass to normal App Handler
        response = await call_next(request)
        response.headers["X-CY02-Sentinel-Score"] = str(req_risk_score)
        response.headers["X-CY02-Latency-MS"] = f"{latency_ms:.2f}"
        return response

# Standard Demo FastAPI App showcasing 2-line integration
if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="E-Commerce Store Protected by CY-02")

    # -------------------------------------------------------------
    # 🔌 PLUG-AND-PLAY INTEGRATION (Just 2 Lines!)
    # -------------------------------------------------------------
    app.add_middleware(CY02SentinelMiddleware, risk_threshold=70.0)

    @app.get("/")
    def home():
        return {"message": "Welcome to the E-Commerce Store!"}

    @app.post("/api/checkout")
    def checkout():
        return {"message": "Checkout successful!"}

    print("🚀 Starting Plug-and-Play Demo Store Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)