"""
predict_tools.py
Model + (varsa) scaler kullanarak tekil/çoklu tahmin yardımcıları.
"""
import pandas as pd
from joblib import load
from src.config import FEATURES_NUM, FEATURES_CAT, MODEL_PATH, SCALER_PATH
from src.predict import predict_df, predict_dict  # src'deki fonksiyonları re-export da edebiliriz

def quick_predict(age: float, height: float, sex):
    """Basit tekil tahmin yardımcı fonksiyonu ('M'/'F' veya 1/0 kabul eder)."""
    sample = {"age": age, "height": height, "sex": sex}
    return predict_dict(sample)

def batch_predict(records: list[dict]) -> pd.Series:
    """Bir dict listesi (kayıt) alır ve tahmin serisi döndürür."""
    df = pd.DataFrame(records)
    preds = predict_df(df)
    return pd.Series(preds, index=df.index, name="prediction")

def load_model_and_scaler():
    """Gerekirse model ve scaler'ı manuel kullanmak için."""
    model = load(MODEL_PATH)
    try:
        scaler = load(SCALER_PATH)
    except Exception:
        scaler = None
    return model, scaler
