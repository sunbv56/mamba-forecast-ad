from sklearn.mixture import GaussianMixture
import numpy as np

class GMMThreshold:
    def __init__(self, n_components=1):
        self.gmm = GaussianMixture(n_components=n_components)
        self.threshold = None

    def fit(self, healthy_anomaly_scores):
        """
        Fit GMM trên tập dữ liệu lành mạnh để tìm phân phối bình thường.
        """
        scores = np.array(healthy_anomaly_scores).reshape(-1, 1)
        self.gmm.fit(scores)
        
        # Ngưỡng có thể là mean + 3*std hoặc tính dựa trên xác suất CDF
        mean = self.gmm.means_[0][0]
        std = np.sqrt(self.gmm.covariances_[0][0][0])
        self.threshold = mean + 3 * std
        return self.threshold

    def is_degraded(self, score):
        if self.threshold is None:
            raise ValueError("Cần gọi fit() trước khi infer.")
        return score > self.threshold
