import pandas as pd
import numpy as np

def detect_credential_stuffing(row):
    """
    High login pressure + high failure rate + multiple unique usernames tried + bot UA.
    """
    score = 0.0
    reasons = []
    
    if row['login_ratio'] > 0.7 and row['failure_pct'] > 0.7:
        score += 0.5
        reasons.append(f"High login failure rate ({row['failure_pct']*100:.0f}%) on /api/login")
        
    if row['unique_usernames'] > 5:
        score += 0.3
        reasons.append(f"Attempted {row['unique_usernames']} distinct usernames in 1 min")
        
    if row['is_bot_ua'] == 1:
        score += 0.2
        reasons.append("Automated script User-Agent detected (python-requests/curl)")
        
    return min(1.0, score), "; ".join(reasons)

def detect_scraping(row):
    """
    High volume + visiting many distinct catalog pages + steady automated timing.
    """
    score = 0.0
    reasons = []
    
    if row['catalog_pages'] > 20:
        score += 0.5
        reasons.append(f"Scraped {row['catalog_pages']} unique product pages in 1 min")
        
    if row['cv_gaps'] < 0.2 and row['req_count'] > 30:
        score += 0.3
        reasons.append(f"Fixed inter-request timing interval (CV={row['cv_gaps']:.2f})")
        
    if row['is_bot_ua'] == 1:
        score += 0.2
        reasons.append("Bot User-Agent string present")
        
    return min(1.0, score), "; ".join(reasons)

def detect_enumeration(row):
    """
    Sequential endpoint ID traversal + high 404/403 failure rate.
    """
    score = 0.0
    reasons = []
    
    if row['seq_id_ratio'] > 0.6:
        score += 0.6
        reasons.append(f"Sequential ID traversal detected ({row['seq_id_ratio']*100:.0f}% consecutive IDs)")
        
    if row['failure_pct'] > 0.6:
        score += 0.4
        reasons.append(f"High resource probes failing with 4xx ({row['failure_pct']*100:.0f}%)")
        
    return min(1.0, score), "; ".join(reasons)

def run_rule_detectors(df_features):
    """
    Applies all heuristic rules and returns max rule score + consolidated evidence.
    """
    if df_features.empty:
        return df_features
        
    df = df_features.copy()
    
    stuff_res = df.apply(detect_credential_stuffing, axis=1)
    scrap_res = df.apply(detect_scraping, axis=1)
    enum_res  = df.apply(detect_enumeration, axis=1)
    
    df['stuff_score'] = [r[0] for r in stuff_res]
    df['stuff_ev']    = [r[1] for r in stuff_res]
    
    df['scrap_score'] = [r[0] for r in scrap_res]
    df['scrap_ev']    = [r[1] for r in scrap_res]
    
    df['enum_score']  = [r[0] for r in enum_res]
    df['enum_ev']     = [r[1] for r in enum_res]
    
    # Highest score across all heuristic rules
    df['rule_max_score'] = df[['stuff_score', 'scrap_score', 'enum_score']].max(axis=1)
    
    def assemble_evidence(row):
        ev_list = []
        for col, ev_col in [('stuff_score', 'stuff_ev'), ('scrap_score', 'scrap_ev'), ('enum_score', 'enum_ev')]:
            if row[col] > 0.3 and row[ev_col]:
                ev_list.append(row[ev_col])
        return " | ".join(ev_list) if ev_list else "Normal heuristic behavior"

    df['rule_evidence'] = df.apply(assemble_evidence, axis=1)
    return df

if __name__ == "__main__":
    print("Running rule-based detectors on window_features.csv...")
    feat_df = pd.read_csv("window_features.csv")
    ruled_df = run_rule_detectors(feat_df)
    
    print("\nRule evaluation complete!")
    print("\nSample High-Risk Heuristic Detections:")
    high_risk = ruled_df[ruled_df['rule_max_score'] > 0.5]
    print(high_risk[['client_id', 'rule_max_score', 'rule_evidence']].head(10))
    
    ruled_df.to_csv("ruled_features.csv", index=False)
    print("\nSaved output to ruled_features.csv!")