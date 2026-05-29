"""
Deney yurutme, loglama ve parametre takibi modulu.

Normal, gurultulu ve unseen senaryolar icin deneyleri calistirir.
Her deney; parametrelerini, sonuclarini ve seed'i kaydeder.
"""
import numpy as np
import logging
import time
from typing import List, Dict, Any
from copy import deepcopy

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Farkli senaryo ve parametre kombinasyonlarinda deneyleri yurutir.

    Desteklenen senaryolar:
        - 'original'  : Ham veri
        - 'noisy'     : Gaussian gurultu ekli veri
        - 'unseen'    : Egitim sozlugunde bulunmayan pattern'larla test
    """

    def __init__(self, config: dict):
        """
        Args:
            config: config.yaml'dan yuklenen sozluk.
        """
        self.config  = config
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def add_gaussian_noise(self, data: np.ndarray,
                           noise_std: float = 0.1,
                           seed: int = 42) -> np.ndarray:
        """
        Veriye Gaussian (normal) gurultu ekler.

        Args:
            data:      Giris verisi.
            noise_std: Gurultu standart sapmasi.
            seed:      Yeniden uretilebilirlik tohumu.

        Returns:
            Gurultulu veri.
        """
        rng   = np.random.default_rng(seed)
        noise = rng.normal(0, noise_std, size=data.shape)
        return data + noise

    # ------------------------------------------------------------------
    def create_unseen_scenario(self, X_test: np.ndarray,
                               fraction: float = 0.2,
                               seed: int = 42) -> np.ndarray:
        """
        Test verisine rastgele yuksek amplitude ekleyerek unseen pattern olusturur.

        Args:
            X_test:   Test verisi.
            fraction: Etkilenecek oranı (0-1).
            seed:     Yeniden uretilebilirlik tohumu.

        Returns:
            Degistirilmis test verisi.
        """
        rng      = np.random.default_rng(seed)
        data     = deepcopy(X_test)
        n        = int(len(data) * fraction)
        indices  = rng.choice(len(data), size=n, replace=False)
        data[indices] += rng.uniform(3.0, 5.0, size=n)
        return data

    # ------------------------------------------------------------------
    def run(self, pipeline, X_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            scenario: str = "original",
            params: dict = None,
            seed: int = 42) -> Dict:
        """
        Tek bir deneyi calistirir ve sonucu kaydeder.

        Args:
            pipeline:  Egitilmis Pipeline nesnesi.
            X_train:   Egitim verisi.
            X_test:    Test verisi.
            y_test:    Test etiketleri.
            scenario:  'original', 'noisy' veya 'unseen'.
            params:    Deney parametreleri sozlugu.
            seed:      Rastgele tohum.

        Returns:
            Deney sonucu sozlugu.
        """
        logger.info("Deney basliyor: senaryo=%s seed=%d", scenario, seed)
        start = time.time()

        # Senaryoya gore test verisini hazirla
        if scenario == "noisy":
            X_eval = self.add_gaussian_noise(X_test, seed=seed)
        elif scenario == "unseen":
            X_eval = self.create_unseen_scenario(X_test, seed=seed)
        else:
            X_eval = X_test

        # Pipeline ile tahmin
        output = pipeline.predict(X_eval)
        elapsed = time.time() - start

        result = {
            "scenario":         scenario,
            "seed":             seed,
            "params":           params or {},
            "path_probability": output["path_probability"],
            "elapsed_sec":      round(elapsed, 3),
        }

        self.results.append(result)
        logger.info("Deney tamamlandi: %.4f saniye", elapsed)
        return result

    # ------------------------------------------------------------------
    def run_multi_seed(self, pipeline, X_train, X_test, y_test,
                       scenario: str = "original",
                       params: dict = None) -> List[Dict]:
        """
        Tüm seed'lerle ayni deneyi tekrarlar.

        Returns:
            Her seed icin sonuc sozlugu listesi.
        """
        seeds   = self.config["training"]["random_seeds"]
        results = []
        for s in seeds:
            res = self.run(pipeline, X_train, X_test, y_test,
                           scenario=scenario, params=params, seed=s)
            results.append(res)
        return results

    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        """
        Tum deney sonuclarinın ortalama ve std'ini hesaplar.

        Returns:
            Ozet istatistikler sozlugu.
        """
        if not self.results:
            return {}
        probs = [r["path_probability"] for r in self.results]
        return {
            "count": len(probs),
            "mean_path_prob": round(float(np.mean(probs)), 6),
            "std_path_prob":  round(float(np.std(probs)), 6),
        }
