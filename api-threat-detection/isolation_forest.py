import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ['req_count', 'failure_pct', 'cv_gaps', 'login_ratio', 'catalog_pages', 'seq_id_ratio', 'is_bot_ua']

class ThreatScoringEngine:
    def __init__(self):
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        
    def fit(self, history_features_df):
        """
        Trains Isolation Forest purely on clean historical window features.
        """
        if history_features_df.empty:
            return
            
        X_train = history_features_df[FEATURE_COLS].fillna(0)
        self.iso_forest.fit(X_train)

    def predict_risk(self, df_features, baseline_model, markov_model, raw_live_df):
        """
        Calculates Final Risk Score = 100 * (0.5*Rules + 0.3*IsoForest + 0.2*Sequence) - LegitDiscount
        """
        if df_features.empty:
            return df_features

        df = df_features.copy()
        X = df[FEATURE_COLS].fillna(0)
        
        # 1. Isolation Forest Anomaly Score (normalized 0.0 to 1.0)
        raw_scores = self.iso_forest.score_samples(X) # Lower score = higher anomaly
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            df['iso_score'] = 1.0 - (raw_scores - min_s) / (max_s - min_s)
        else:
            df['iso_score'] = 0.0
        
        # 2. Sequence Anomaly Score via Markov Chain
        seq_scores = []
        for _, row in df.iterrows():
            cid = row['client_id']
            client_reqs = raw_live_df[raw_live_df['client_id'] == cid]
            eps = client_reqs['endpoint'].tolist()
            seq_scores.append(markov_model.get_transition_score(eps))
            
        df['sequence_anomaly'] = seq_scores
        
        # 3. Z-Score against client's historical baseline
        df['z_score'] = df.apply(lambda r: baseline_model.compute_zscore(r['client_id'], r['req_count']), axis=1)
        
        # 4. Ensemble Aggregation Formula
        raw_risk = 100.0 * (
            0.5 * df['rule_max_score'] + 
            0.3 * df['iso_score'] + 
            0.2 * df['sequence_anomaly']
        )
        
        # 5. Legitimate Heavy B2B Partner Discount (Prevent False Positives)
        legit_discount = df.apply(
            lambda r: 40.0 if (r['failure_pct'] < 0.01 and r['is_bot_ua'] == 0 and r['unique_usernames'] == 0) else 0.0,
            axis=1
        )
        
        final_risk = np.clip(raw_risk - legit_discount, 0.0, 100.0)
        df['final_risk_score'] = np.round(final_risk, 1)
        
        # Categorize Risk Level
        df['risk_category'] = pd.cut(
            df['final_risk_score'],
            bins=[-1.0, 30.0, 70.0, 100.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        return df


if __name__ == "__main__":
    from feature_extraction import extract_window_features
    from rule_detectors import run_rule_detectors
    from baseline_models import AdaptiveBaseline
    from sequence_model import MarkovSequenceDetector

    print("Initializing pipeline and training models on clean history...")
    history_df = pd.read_csv("data/history_logs.csv")
    history_feats = extract_window_features(history_df, window_minutes=1)

    baseline = AdaptiveBaseline()
    baseline.fit_history(history_feats)

    markov = MarkovSequenceDetector()
    markov.fit(history_df)

    engine = ThreatScoringEngine()
    engine.fit(history_feats)

    print("Processing live evaluation logs...")
    live_df = pd.read_csv("data/live_logs.csv")
    live_feats = extract_window_features(live_df, window_minutes=1)
    ruled_feats = run_rule_detectors(live_feats)

    final_df = engine.predict_risk(ruled_feats, baseline, markov, live_df)

    print("\nScoring Complete! Sample High-Risk Alerts:")
    print(final_df[final_df['final_risk_score'] > 70][['client_id', 'req_count', 'final_risk_score', 'risk_category', 'rule_evidence']].head(10))

    # Verify partner_acme status
    acme_max_risk = final_df[final_df['client_id'] == 'partner_acme']['final_risk_score'].max()
    print(f"\nPartner Acme Max Risk Score: {acme_max_risk} (Expected: Low/Green < 30)")

    final_df.to_csv("final_risk_scores.csv", index=False)
    print("Saved complete pipeline results to final_risk_scores.csv!")