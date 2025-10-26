"""
pipeline.py
Notebook'tan yüksek seviyeli adım çağrıları: indir → hazırla → eğit.
Ayrıca processed veriyi ve metrikleri yüklemek için yardımcılar.
"""
from pathlib import Path
import json
import pandas as pd

# src modüllerini kullanıyoruz
from src.download import main as download_main
from src.prepare import main as prepare_main
from src.train import main as train_main
from src.config import TRAIN_DATA, TEST_DATA, METRICS_PATH, RAW_DATA

def run_download():
    """Kaggle datasetini indirip data/raw altına yazar."""
    download_main()

def run_prepare():
    """Temizlik + split + ölçekleme (ve scaler kaydı)."""
    prepare_main()

def run_train():
    """Linear Regression eğit + metrikleri kaydet."""
    train_main()

def load_raw(n=5) -> pd.DataFrame:
    """Ham CSV'den ilk n satırı oku (hızlı bakış)."""
    return pd.read_csv(RAW_DATA).head(n)

def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """processed train/test dataframes döndür."""
    train = pd.read_csv(TRAIN_DATA)
    test  = pd.read_csv(TEST_DATA)
    return train, test

def load_metrics() -> dict:
    """reports/metrics.json içeriğini oku."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
