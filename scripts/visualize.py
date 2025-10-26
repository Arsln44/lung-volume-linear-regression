"""
visualize.py
Matplotlib ile temel görselleştirmeler.
"""
from typing import Sequence, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms(df: pd.DataFrame, cols: Sequence[str], bins: int = 30):
    """Seçilen kolonlar için tek tek histogram çizer."""
    for col in cols:
        if col not in df.columns: 
            print(f"[viz] atlandı: {col} df'de yok")
            continue
        df[col].hist(bins=bins)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.show()

def plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Gerçek vs Tahmin"):
    """Gerçek vs Tahmin scatter grafiği (diagnostic)."""
    plt.scatter(y_true, y_pred, s=18)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Gerçek")
    plt.ylabel("Tahmin")
    plt.title(title)
    plt.show()

def plot_corr_heatmap(df: pd.DataFrame, cols: Optional[Sequence[str]] = None):
    """Korelasyon ısı haritası (yalnızca sayısal kolonlar)."""
    data = df[cols] if cols else df.select_dtypes(include=["number"])
    corr = data.corr(numeric_only=True)
    plt.imshow(corr, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right")
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.title("Korelasyon Isı Haritası")
    plt.tight_layout()
    plt.show()
