"""
explore.py
Hızlı EDA yardımcıları: özet, eksik değerler, temel istatistik.
"""
import pandas as pd

def info_summary(df: pd.DataFrame) -> dict:
    """df.info() eşdeğeri özet; notebook'ta okunması kolay bir sözlük döndürür."""
    dtypes = df.dtypes.astype(str).to_dict()
    rows, cols = df.shape
    return {"shape": (rows, cols), "dtypes": dtypes}

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik değer sayıları ve oranları."""
    miss = df.isna().sum()
    ratio = (miss / len(df)).round(4)
    rep = pd.DataFrame({"missing": miss, "ratio": ratio})
    return rep[rep["missing"] > 0].sort_values("missing", ascending=False)

def describe_numeric(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Sayısal kolonlar için describe() çıktısı (kısaltılmış, sürüm uyumlu)."""
    # numeric_only parametresi olmayan eski sürümlerle uyumlu hale getirdik
    numeric_df = df.select_dtypes(include=["number"])
    desc = numeric_df.describe().T
    return desc.head(top_n)
