"""
Bu dosyada eğitilmiş model ve varsa scaler kullanarak 
yeni veriler üzerinde tahminler yapar.
"""
from __future__ import annotations
import pandas as pd
from joblib import load
from .config import (
    MODEL_PATH, 
    SCALER_PATH,
    FEATURES_NUM,
    FEATURES_CAT
)

def _maybe_load_scaler(): # Varsa scaler'ı yükler
    try:
        return load(SCALER_PATH)
    except Exception:
        return None
    
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame: # Gerekli sütunların varlığını kontrol eder
    needed = FEATURES_NUM + FEATURES_CAT
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")
    return df[needed].copy()

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame: # Sütun tip uyumluluğunu sağlar
    num_cols = list(FEATURES_NUM) if FEATURES_NUM else []
    cat_cols = list(FEATURES_CAT) if FEATURES_CAT else []
    # Kategorik sütunlar için
    for col in FEATURES_CAT:
        if df[col].dtype == 'object':
            # 'M' 'm' 'F' 'f' gibi varyasyonları düzelt
            s = df[col].astype(str).str.upper().str.strip()
            df[col] = s.map({"M": 1 , "F": 0})
    # Numerik sütunlar için
    for col in FEATURES_NUM:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    if df[num_cols + cat_cols].isna().any().any():
        bad = df[num_cols + cat_cols].isna().sum()
        raise ValueError(f"Girdi içinde NaN oluştu. Kontrol et: {bad.to_dict()}")
    return df

def predict_df(df: pd.DataFrame): # Bir DataFrame alır varsa scaler ile ölçeklendirir ve tahmin yapar
    df = _ensure_columns(df)
    df = _coerce_types(df)

    scaler = _maybe_load_scaler()
    num_cols = list(FEATURES_NUM) if FEATURES_NUM else []
    if scaler is not None and num_cols:
        df.loc[:, num_cols] = scaler.transform(df[num_cols]) # df.loc DataFrame de etiket tabanlı erişim sağlar örn: df.loc['age']

    model = load(MODEL_PATH)
    preds = model.predict(df)
    return preds

def predict_dict(sample: dict): # Bir veriyi sözlük olarak alır ve tahmin yapar
    df = pd.DataFrame([sample]) 
    return float(predict_df(df)[0])

def main():
    example_data = {"age": 24, "height": 178, "sex": "M"}
    y_pred = predict_dict(example_data)
    print(f"[predict] input={example_data} -> prediction={y_pred}")

if __name__ == "__main__":
    main()
