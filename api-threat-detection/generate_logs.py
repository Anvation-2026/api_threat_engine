import numpy as np, pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)  # Seed for 100% reproducibility

START = datetime(2026, 10, 8, 9, 0, 0)   # 2-hour baseline history
LIVE  = START + timedelta(hours=2)       # 1-hour live test stream

rows = []

BROWSERS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
]
BOT_UA = "python-requests/2.31.0"

def rand_ip():
    return f"{rng.integers(20,220)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,255)}"

def log(t, client, ip, method, ep, status, ua, label, username=None, sid=None):
    rows.append(dict(
        timestamp=t, client_id=client, ip=ip, session_id=sid or client,
        method=method, endpoint=ep, username=username, status=status,
        resp_bytes=int(rng.normal(900, 200)) if status == 200 else int(rng.normal(300, 40)),
        latency_ms=max(5, int(rng.normal(90, 25))),
        user_agent=ua, label=label
    ))

# ---------- NORMAL USER ----------
def normal_user(i, start, speed=1.0, label="normal"):
    t, cid, ip, ua = start, f"user_{label}_{i}", rand_ip(), rng.choice(BROWSERS)
    def step(method, ep, status=200, username=None):
        nonlocal t
        # Human pauses are log-normal/exponential (high variance)
        t += timedelta(seconds=(rng.exponential(8) + 1) * speed)
        if rng.random() < 0.04: status = 404
        elif rng.random() < 0.01: status = 401 # Occasional session expiry
        log(t, cid, ip, method, ep, status, ua, label, username, sid=cid + "_s")

    step("POST", "/api/login", 200 if rng.random() > 0.05 else 401, username=f"user{i}")
    for _ in range(rng.integers(2, 8)):
        step("GET", f"/api/products?page={rng.integers(1, 20)}")
        if rng.random() < 0.6: 
            step("GET", f"/api/products/{rng.integers(1, 500)}")
    if rng.random() < 0.4:
        step("POST", "/api/cart")
        if rng.random() < 0.6: 
            step("POST", "/api/checkout")

# ---------- LEGIT HEAVY CLIENT ----------
def legit_heavy(start, minutes, per_hour=5000):
    gap, t = 3600 / per_hour, start
    end = start + timedelta(minutes=minutes)
    while t < end:
        t += timedelta(seconds=gap * rng.uniform(0.85, 1.15))
        ep = rng.choice(["/api/orders/sync", "/api/inventory"])
        st = 200 if rng.random() > 0.002 else 500
        log(t, "partner_acme", "52.66.10.20", "POST", ep, st, "AcmeSync/3.2", "legit_heavy")

# ---------- CREDENTIAL STUFFER ----------
def stuffer(start, ip, n=400):
    t = start
    for k in range(n):
        # Extremely tight timing regularity (bot indicator)
        t += timedelta(seconds=0.5 + rng.normal(0, 0.01))
        st = 200 if rng.random() < 0.03 else 401
        log(t, ip, ip, "POST", "/api/login", st, BOT_UA, "credential_stuffing",
            username=f"user{rng.integers(1, 100000)}")

# ---------- LOW AND SLOW ----------
def low_and_slow(start, n_ips=30, tries=4):
    for _ in range(n_ips):
        ip = rand_ip()
        # Blends in by using clean browser UAs instead of python-requests
        ua = rng.choice(BROWSERS)
        for _ in range(tries):
            t = start + timedelta(seconds=float(rng.uniform(0, 3600)))
            log(t, ip, ip, "POST", "/api/login", 401, ua, "low_and_slow",
                username=f"user{rng.integers(1, 100000)}")

# ---------- SCRAPER ----------
def scraper(start, ip, pages=500):
    t = start
    for p in range(1, pages + 1):
        t += timedelta(seconds=0.3 * rng.uniform(0.98, 1.02))
        log(t, ip, ip, "GET", f"/api/products?page={p}", 200, BOT_UA, "scraper")

# ---------- ENUMERATOR ----------
def enumerator(start, ip, n=600):
    t = start
    for uid in range(1001, 1001 + n):
        t += timedelta(seconds=0.4 * rng.uniform(0.9, 1.1))
        st = rng.choice([404, 403, 200], p=[0.85, 0.10, 0.05])
        log(t, ip, ip, "GET", f"/api/users/{uid}", int(st), BOT_UA, "enumeration")

# ---------- ABNORMAL SEQUENCE ----------
def abnormal(i, start):
    t, cid, ip, ua = start, f"odd_{i}", rand_ip(), rng.choice(BROWSERS)
    for ep in rng.choice(["/api/checkout", "/api/admin/export", "/api/cart"], size=rng.integers(3, 6)):
        t += timedelta(seconds=float(rng.uniform(1, 5)))
        log(t, cid, ip, "GET", ep, 403, ua, "abnormal_sequence")

# ================= BUILD DATASET =================
# HISTORY PHASE (2 Hours - Purely Clean)
for i in range(300): 
    normal_user(i, START + timedelta(seconds=float(rng.uniform(0, 7200))))
legit_heavy(START, minutes=120)

# LIVE PHASE (1 Hour - Attacks + Traffic)
for i in range(150): 
    normal_user(1000 + i, LIVE + timedelta(seconds=float(rng.uniform(0, 3600))))
legit_heavy(LIVE, minutes=60)
for i in range(300):
    normal_user(i, LIVE + timedelta(minutes=30, seconds=float(rng.uniform(0, 300))), speed=0.3, label="flash_sale")

stuffer(LIVE + timedelta(minutes=10), "185.220.5.9")
low_and_slow(LIVE)
scraper(LIVE + timedelta(minutes=20), "45.33.12.7")
enumerator(LIVE + timedelta(minutes=40), "91.108.4.3")
for i in range(20): 
    abnormal(i, LIVE + timedelta(minutes=float(rng.uniform(5, 55))))

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

# Split and export for easy evaluation downstream
df_history = df[df['timestamp'] < LIVE].reset_index(drop=True)
df_live = df[df['timestamp'] >= LIVE].reset_index(drop=True)

df.to_csv("api_logs.csv", index=False)
df_history.to_csv("history_logs.csv", index=False)
df_live.to_csv("live_logs.csv", index=False)

print("--- GENERATION COMPLETE ---")
print(df['label'].value_counts())