import httpx
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI(title="Target E-Commerce Store")

CY02_INGEST_URL = "http://127.0.0.1:9000/api/v1/logs"

@app.middleware("http")
async def send_logs_to_cy02(request: Request, call_next):
    response = await call_next(request)
    
    payload = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": request.headers.get("X-Client-ID", f"client_{request.client.host}"),
        "ip": request.client.host,
        "endpoint": request.url.path,
        "status": response.status_code,
        "user_agent": request.headers.get("User-Agent", "Unknown"),
        "username": "customer_1"
    }

    # Forward log event to CY-02 Sentinel asynchronously
    async with httpx.AsyncClient() as client:
        try:
            await client.post(CY02_INGEST_URL, json=payload, timeout=0.5)
        except Exception:
            pass

    return response

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shopify / E-Commerce Store</title>
        <style>
            body { font-family: Arial; background: #0d1117; color: #fff; padding: 40px; text-align: center; }
            .card { background: #161b22; border: 1px solid #30363d; padding: 20px; margin: 15px auto; width: 320px; border-radius: 8px; }
            button { background: #238636; color: white; border: none; padding: 10px 18px; border-radius: 5px; cursor: pointer; margin: 5px; }
            button.danger { background: #da3633; }
        </style>
    </head>
    <body>
        <h1>🛒 Target E-Commerce Web Application</h1>
        <p>Interactions send live API telemetry to <b>CY-02 Sentinel</b>.</p>
        
        <div class="card">
            <h3>📦 Product Catalog</h3>
            <button onclick="fetch('/api/products')">View Items</button>
            <button onclick="fetch('/api/checkout', {method: 'POST'})">Buy Item</button>
        </div>

        <div class="card">
            <h3>🔐 Authentication Portal</h3>
            <button onclick="fetch('/api/login', {method: 'POST'})">Normal Login</button>
            <button class="danger" onclick="triggerBot()">Simulate Bot Attack</button>
        </div>

        <script>
            function triggerBot() {
                for(let i=0; i<8; i++) {
                    fetch('/api/login', {
                        method: 'POST',
                        headers: {'User-Agent': 'Python-requests/Bot-v1', 'X-Client-ID': 'attacker_bot_99'}
                    });
                }
                alert('Fired 8 bot login requests to CY-02!');
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/products")
def get_products(): return {"data": ["Laptop", "Smartphone"]}

@app.post("/api/login")
def login(): return {"status": "login_failed"}

@app.post("/api/checkout")
def checkout(): return {"status": "order_success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    