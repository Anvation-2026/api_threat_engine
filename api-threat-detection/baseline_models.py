import pandas as pd
import numpy as np

class AdaptiveBaseline:
    def __init__(self):
        self.client_stats = {}
        self.global_stats = {}

    def fit_history(self, history_features_df):
        """
        Learns mean and std dev of request rates per client from clean history logs.
        """
        if history_features_df.empty:
            return

        # Global fallback baseline for new/unseen clients
        self.global_stats = {
            'mean_req': history_features_df['req_count'].mean(),
            'std_req': max(history_features_df['req_count'].std(), 1.0)
        }
        
        # Per-client baselines
        for client_id, group in history_features_df.groupby('client_id'):
            self.client_stats[client_id] = {
                'mean_req': group['req_count'].mean(),
                'std_req': max(group['req_count'].std(), 1.0)
            }

    def compute_zscore(self, client_id, current_req_count):
        """
        Computes Z-Score: (current - mean) / std_dev against client's OWN history.
        """
        if client_id in self.client_stats:
            stats = self.client_stats[client_id]
        else:
            stats = self.global_stats
            
        z = (current_req_count - stats['mean_req']) / stats['std_req']
        return round(float(max(0.0, z)), 2)


if __name__ == "__main__":
    from feature_extraction import extract_window_features
    
    print("Fitting Adaptive Baselines on data/history_logs.csv...")
    history_df = pd.read_csv("data/history_logs.csv")
    history_feats = extract_window_features(history_df, window_minutes=1)
    
    baseline_model = AdaptiveBaseline()
    baseline_model.fit_history(history_feats)
    
    print(f"Learned historical baselines for {len(baseline_model.client_stats)} distinct clients.")
    
    # Test on partner_acme vs normal user
    acme_z = baseline_model.compute_zscore("partner_acme", current_req_count=85)
    print(f"Z-Score for 'partner_acme' at 85 req/min: {acme_z} (Expected: Low because Acme is normally heavy)")
    
    new_user_z = baseline_model.compute_zscore("user_normal_999", current_req_count=85)
    print(f"Z-Score for 'user_normal_999' at 85 req/min: {new_user_z} (Expected: High anomaly)")