import pandas as pd
import numpy as np

class MarkovSequenceDetector:
    def __init__(self, laplace_alpha=1.0):
        self.alpha = laplace_alpha
        self.transitions = {}
        self.endpoints = set()

    def fit(self, raw_logs_df):
        """
        Builds endpoint transition probability matrix from clean history logs.
        """
        if raw_logs_df.empty:
            return

        df = raw_logs_df.sort_values(['client_id', 'timestamp'])
        
        # Collect transition counts between consecutive endpoints
        for client_id, group in df.groupby('client_id'):
            eps = group['endpoint'].tolist()
            for i in range(len(eps) - 1):
                src, dst = eps[i], eps[i+1]
                self.endpoints.add(src)
                self.endpoints.add(dst)
                if src not in self.transitions:
                    self.transitions[src] = {}
                self.transitions[src][dst] = self.transitions[src].get(dst, 0) + 1

    def get_transition_score(self, sequence):
        """
        Evaluates a sequence of endpoints.
        Returns anomaly score between 0.0 (very normal sequence) and 1.0 (abnormal jump).
        """
        if len(sequence) < 2:
            return 0.0
            
        log_prob = 0.0
        vocab_size = max(len(self.endpoints), 10)
        
        for i in range(len(sequence) - 1):
            src, dst = sequence[i], sequence[i+1]
            src_counts = self.transitions.get(src, {})
            total_out = sum(src_counts.values())
            trans_count = src_counts.get(dst, 0)
            
            # Laplace smoothing formula to prevent division by zero
            prob = (trans_count + self.alpha) / (total_out + self.alpha * vocab_size)
            log_prob += np.log(prob)
            
        avg_log_prob = log_prob / (len(sequence) - 1)
        
        # Scale log-likelihood to a normalized 0.0 - 1.0 anomaly score
        anomaly_score = 1.0 - (1.0 / (1.0 + np.exp(-(avg_log_prob + 3.0))))
        return round(float(anomaly_score), 4)


if __name__ == "__main__":
    print("Training Markov Sequence Detector on data/history_logs.csv...")
    history_df = pd.read_csv("data/history_logs.csv")
    
    markov = MarkovSequenceDetector()
    markov.fit(history_df)
    
    # Test 1: Normal sequence
    normal_seq = ["/api/login", "/api/products?page=1", "/api/products/12", "/api/cart"]
    score_normal = markov.get_transition_score(normal_seq)
    print(f"\nSequence Score for Normal path: {score_normal} (Expected: Low anomaly)")
    
    # Test 2: Abnormal sequence
    abnormal_seq = ["/api/checkout", "/api/admin/export", "/api/cart"]
    score_abnormal = markov.get_transition_score(abnormal_seq)
    print(f"Sequence Score for Abnormal path: {score_abnormal} (Expected: High anomaly)")