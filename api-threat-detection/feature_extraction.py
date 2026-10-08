import pandas as pd
import numpy as np

BOT_KEYWORDS = ["python", "curl", "requests", "bot", "scraper", "headless"]

def extract_window_features(df, window_minutes=1):
    """
    Groups raw API logs into time windows per client_id (or IP) 
    and computes behavioral features for rules and ML models.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Floor timestamps to window intervals (e.g., 1-minute blocks)
    df['window_time'] = df['timestamp'].dt.floor(f'{window_minutes}min')
    
    features = []
    
    # Group by Client ID and Time Window
    grouped = df.groupby(['client_id', 'window_time'])
    
    for (client_id, win_time), group in grouped:
        req_count = len(group)
        if req_count == 0:
            continue
            
        # 1. Failure Percentage
        failures = group['status'].apply(lambda s: 1 if s >= 400 else 0).sum()
        failure_pct = failures / req_count
        
        # 2. Timing Regularity (Coefficient of Variation of time gaps)
        if req_count > 1:
            gaps = group['timestamp'].diff().dt.total_seconds().dropna()
            mean_gap = gaps.mean()
            std_gap = gaps.std()
            cv_gaps = (std_gap / mean_gap) if mean_gap > 0 and not np.isnan(std_gap) else 0.0
        else:
            cv_gaps = 1.0  # Neutral default for single request
            
        # 3. Unique Usernames
        unique_usernames = group['username'].dropna().nunique()
        
        # 4. Login Pressure
        login_reqs = group['endpoint'].apply(lambda ep: 1 if ep == '/api/login' else 0).sum()
        login_ratio = login_reqs / req_count
        
        # 5. Catalog Coverage (unique page numbers)
        catalog_pages = group['endpoint'].apply(
            lambda ep: ep.split('page=')[1] if 'page=' in ep else None
        ).dropna().nunique()
        
        # 6. Sequential ID Ratio (detecting /api/users/1001, 1002...)
        user_ids = group['endpoint'].apply(
            lambda ep: int(ep.split('/')[-1]) if ep.startswith('/api/users/') and ep.split('/')[-1].isdigit() else None
        ).dropna().tolist()
        
        if len(user_ids) > 1:
            diffs = np.diff(user_ids)
            sequential_count = np.sum(diffs == 1)
            seq_id_ratio = sequential_count / len(diffs)
        else:
            seq_id_ratio = 0.0
            
        # 7. User Agent Flag
        sample_ua = str(group['user_agent'].iloc[0]).lower()
        is_bot_ua = 1 if any(b in sample_ua for b in BOT_KEYWORDS) else 0
        
        # 8. Primary IP and Label (label kept purely for final model evaluation)
        primary_ip = group['ip'].iloc[0]
        primary_label = group['label'].mode()[0] if 'label' in group.columns else 'unknown'
        
        features.append({
            'window_time': win_time,
            'client_id': client_id,
            'ip': primary_ip,
            'req_count': req_count,
            'failure_pct': round(failure_pct, 4),
            'cv_gaps': round(cv_gaps, 4),
            'unique_usernames': unique_usernames,
            'login_ratio': round(login_ratio, 4),
            'catalog_pages': catalog_pages,
            'seq_id_ratio': round(seq_id_ratio, 4),
            'is_bot_ua': is_bot_ua,
            'label': primary_label
        })
        
    return pd.DataFrame(features)

if __name__ == "__main__":
    print("Extracting windowed features from data/live_logs.csv...")
    live_df = pd.read_csv("data/live_logs.csv")
    feat_df = extract_window_features(live_df, window_minutes=1)
    
    print(f"\nSuccessfully extracted {len(feat_df)} feature window records.")
    print("\nSample Output:")
    print(feat_df[['client_id', 'req_count', 'failure_pct', 'cv_gaps', 'login_ratio', 'label']].head(10))
    
    feat_df.to_csv("window_features.csv", index=False)
    print("\nSaved extracted features to window_features.csv!")