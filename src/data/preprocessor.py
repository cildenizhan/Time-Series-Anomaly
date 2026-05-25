from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class TimeSeriesPreprocessor:
    def __init__(self, use_pca=False):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1) if use_pca else None
        self.is_fitted = False
        
    def fit_transform(self, X):
        """Normalizasyon ve PCA sadece train verisi uzerinde fit edilmelidir."""
        X_scaled = self.scaler.fit_transform(X)
        if self.pca:
            X_scaled = self.pca.fit_transform(X_scaled)
        self.is_fitted = True
        return X_scaled
        
    def transform(self, X):
        """Validation ve Test verisine ayni donusum uygulanir."""
        if not self.is_fitted:
            raise ValueError("Preprocessor henuz fit edilmedi!")
        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)
        return X_scaled
