"""
Gorsellestirme modulu.
Otomata durum diyagrami, gecis olasiligi heatmap ve performans grafikleri.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional


def plot_transition_heatmap(matrix: np.ndarray, labels: List[str],
                            title: str = "Gecis Olasiligi Heatmap",
                            save_path: Optional[str] = None) -> None:
    """
    Gecis olasilik matrisini heatmap olarak gorseller.

    Args:
        matrix:    (N, N) gecis olasilik matrisi.
        labels:    Durum etiketleri listesi.
        title:     Grafik basligi.
        save_path: Kaydedilecek dosya yolu (None ise gosterilir).
    """
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Gecis Olasiligi")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Hedef Durum")
    ax.set_ylabel("Kaynak Durum")
    ax.set_title(title)

    # Hucre degerleri
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm: List[List[int]],
                          labels: List[str] = None,
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None) -> None:
    """
    Confusion matrix'i gorseller.

    Args:
        cm:        2x2 confusion matrix (liste veya ndarray).
        labels:    Sinif etiketleri (varsayilan: ['Normal', 'Anomaly']).
        title:     Grafik basligi.
        save_path: Kaydedilecek dosya yolu.
    """
    cm_arr = np.array(cm)
    labels = labels or ["Normal", "Anomali"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Oranges")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gercek")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_path_probabilities(probs: List[float],
                            threshold: float = 0.05,
                            title: str = "Yol Olasiliklari",
                            save_path: Optional[str] = None) -> None:
    """
    Zaman adimlarindaki yol olasiliklerini cizgi grafik olarak gorseller.

    Args:
        probs:     Her zaman adimi icin yol olasiligi listesi.
        threshold: Anomali esigi (kirmizi yatay cizgi).
        title:     Grafik basligi.
        save_path: Kaydedilecek dosya yolu.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(probs))
    ax.plot(x, probs, color="steelblue", linewidth=1.2, label="Yol Olasiligi")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"Esik ({threshold})")
    ax.fill_between(x, probs, threshold,
                    where=[p < threshold for p in probs],
                    color="red", alpha=0.15, label="Anomali Bolgesi")
    ax.set_xlabel("Zaman Adimi")
    ax.set_ylabel("P(dizi)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
