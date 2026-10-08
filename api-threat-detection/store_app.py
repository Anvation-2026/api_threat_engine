import os
import httpx
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Target E-Commerce Store")

# Enable full CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CY02_INGEST_URL = "http://127.0.0.1:9000/api/v1/logs"
LIVE_LOGS_CSV = "data/live_logs.csv"

def append_directly_to_csv(payload: dict):
    """Appends incoming logs using the exact 8-column format expected by CY-02."""
    try:
        os.makedirs("data", exist_ok=True)
        
        new_log = {
            "timestamp": payload["timestamp"],
            "client_id": payload["client_id"],
            "ip": payload["ip"],
            "endpoint": payload["endpoint"],
            "status_code": payload["status"],
            "user_agent": payload["user_agent"],
            "username": payload["username"],
            "label": "credential_stuffing" if "attacker" in payload["client_id"] or "bot" in payload["user_agent"].lower() else "normal"
        }
        
        df_new = pd.DataFrame([new_log])
        file_exists = os.path.exists(LIVE_LOGS_CSV)
        df_new.to_csv(LIVE_LOGS_CSV, mode='a', header=not file_exists, index=False)
    except Exception:
        pass

@app.middleware("http")
async def send_logs_to_cy02(request: Request, call_next):
    # Skip logging root HTML page request or favicon
    if request.url.path in ["/", "/favicon.ico"]:
        return await call_next(request)

    response = await call_next(request)
    
    client_host = request.client.host if request.client else "127.0.0.1"
    payload = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": request.headers.get("X-Client-ID", f"client_{client_host}"),
        "ip": client_host,
        "endpoint": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "user_agent": request.headers.get("User-Agent", "Unknown"),
        "username": request.headers.get("X-Username", "customer_1")
    }

    # Direct CSV append guarantees Streamlit updates instantly
    append_directly_to_csv(payload)

    # Forward log event asynchronously to CY-02 ingest API
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
            body { font-family: Arial, sans-serif; background: #0d1117; color: #fff; padding: 30px; text-align: center; }
            .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
            .card { background: #161b22; border: 1px solid #30363d; padding: 20px; width: 380px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); text-align: left; }
            .card h3 { margin-top: 0; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
            button { background: #238636; color: white; border: none; padding: 8px 14px; border-radius: 5px; cursor: pointer; font-weight: bold; margin-top: 5px; }
            button:hover { opacity: 0.85; }
            button.danger { background: #da3633; width: 100%; margin-top: 10px; }
            button.pay { background: #8957e5; width: 100%; margin-top: 10px; }
            input { background: #0d1117; border: 1px solid #30363d; color: white; padding: 8px; border-radius: 5px; width: 90%; margin-bottom: 10px; }
            .item-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px dashed #30363d; }
            #log-output { background: #161b22; border: 1px solid #30363d; padding: 14px; margin: 25px auto; width: 800px; border-radius: 6px; font-family: monospace; color: #3fb950; text-align: left; font-size: 0.85rem; min-height: 50px; }
        </style>
    </head>
    <body>
        <h1>🛒 Target E-Commerce Web Application</h1>
        <p>Interactions send live API telemetry to <b>CY-02 Sentinel</b>.</p>
        
        <div class="container">
            <!-- Product Catalog Card (5 Items) -->
            <div class="card">
                <h3>📦 Product Catalog (5 Items)</h3>
                <div class="item-row">
                    <span>💻 Laptop ($1,200)</span>
                    <button onclick="addToCart('Laptop', 1200)">Add</button>
                </div>
                <div class="item-row">
                    <span>📱 Smartphone ($800)</span>
                    <button onclick="addToCart('Smartphone', 800)">Add</button>
                </div>
                <div class="item-row">
                    <span>🎧 Headphones ($150)</span>
                    <button onclick="addToCart('Headphones', 150)">Add</button>
                </div>
                <div class="item-row">
                    <span>⌨️ Keyboard ($100)</span>
                    <button onclick="addToCart('Keyboard', 100)">Add</button>
                </div>
                <div class="item-row">
                    <span>⌚ Smart Watch ($250)</span>
                    <button onclick="addToCart('Smart Watch', 250)">Add</button>
                </div>
                <hr style="border-color: #30363d; margin-top: 15px;">
                <div><b>Cart Total: $<span id="cart-total">0</span></b></div>
                <button class="pay" onclick="payTotalBill()">💳 Pay Total Bill</button>
            </div>

            <!-- Authentication Portal Card -->
            <div class="card">
                <h3>🔐 Authentication Portal</h3>
                <p style="font-size: 0.8rem; color: #8b949e;">Test Acc: <b>admin</b> / <b>password123</b></p>
                <label>Username:</label>
                <input type="text" id="username" value="admin">
                <label>Password:</label>
                <input type="password" id="password" value="password123">
                <button style="width: 100%;" onclick="performLogin()">Login</button>

                <hr style="border-color: #30363d; margin: 20px 0 10px 0;">
                <button class="danger" onclick="triggerBot()">⚡ Simulate High-Volume Bot Attack (12 Reqs)</button>
            </div>
        </div>

        <div id="log-output">Status: Ready for interactions.</div>

        <script>
            let currentCartTotal = 0;
            let cartItems = [];

            function addToCart(itemName, price) {
                cartItems.push(itemName);
                currentCartTotal += price;
                document.getElementById('cart-total').innerText = currentCartTotal;
                
                fetch('/api/products')
                    .then(res => res.json())
                    .then(data => {
                        document.getElementById('log-output').innerText = `Added ${itemName} to cart. Item catalog updated.`;
                    });
            }

            async function payTotalBill() {
                const output = document.getElementById('log-output');
                if (currentCartTotal === 0) {
                    output.innerText = '⚠️ Cart is empty! Add items to cart before paying.';
                    return;
                }
                output.innerText = `Processing checkout for total: $${currentCartTotal}...`;
                try {
                    const res = await fetch('/api/checkout', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ amount: currentCartTotal, items: cartItems })
                    });
                    const data = await res.json();
                    output.innerText = `HTTP ${res.status} OK\nPayment Successful! Order ID: ${data.order_id} | Total Paid: $${currentCartTotal}`;
                    currentCartTotal = 0;
                    cartItems = [];
                    document.getElementById('cart-total').innerText = 0;
                } catch(e) {
                    output.innerText = `Error: ${e.message}`;
                }
            }

            async function performLogin() {
                const output = document.getElementById('log-output');
                const u = document.getElementById('username').value;
                const p = document.getElementById('password').value;

                output.innerText = `Authenticating user '${u}'...`;
                try {
                    const res = await fetch('/api/login', {
                        method: 'POST',
                        headers: { 
                            'Content-Type': 'application/json',
                            'X-Username': u
                        },
                        body: JSON.stringify({ username: u, password: p })
                    });
                    const data = await res.json();
                    output.innerText = `HTTP ${res.status} ${res.statusText}\nStatus: ${data.status} | Details: ${data.message || data.reason}`;
                } catch(e) {
                    output.innerText = `Error: ${e.message}`;
                }
            }

            async function triggerBot() {
                const output = document.getElementById('log-output');
                output.innerText = 'Firing 12 automated bot attack requests...';
                
                for (let i = 1; i <= 12; i++) {
                    await fetch('/api/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'User-Agent': 'Python-requests/Bot-v1', 
                            'X-Client-ID': `attacker_bot_${i}`,
                            'X-Username': `victim_user_${i}`
                        },
                        body: JSON.stringify({ username: `victim_user_${i}`, password: 'invalid_password_123' })
                    });
                    output.innerText = `Fired failed bot login ${i}/12 to CY-02...`;
                    await new Promise(r => setTimeout(r, 60));
                }

                output.innerText = '🚨 Fired 12 high-failure bot attack requests! Risk score will reflect > 70 on Streamlit dashboard.';
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/products")
def get_products():
    return {
        "items": [
            {"id": 1, "name": "Laptop", "price": 1200},
            {"id": 2, "name": "Smartphone", "price": 800},
            {"id": 3, "name": "Headphones", "price": 150},
            {"id": 4, "name": "Keyboard", "price": 100},
            {"id": 5, "name": "Smart Watch", "price": 250}
        ]
    }

@app.post("/api/login")
async def login(request: Request):
    try:
        body = await request.json()
        username = body.get("username", "")
        password = body.get("password", "")
        
        if username == "admin" and password == "password123":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "login_success", "message": "Authenticated successfully"}
            )
        
        # Returns HTTP 401 Unauthorized for failed logins so failure rate reaches 100%
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"status": "login_failed", "reason": "Invalid credentials"}
        )
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "login_failed", "reason": "Malformed payload"}
        )

@app.post("/api/checkout")
def checkout():
    return {"status": "order_success", "order_id": "ORD-2026-99A"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("store_app:app", host="127.0.0.1", port=8000, reload=True)