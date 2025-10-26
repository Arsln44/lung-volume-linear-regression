"""
prepare.py dosyası ile hazırlanmış verilerle Linear Regression modelini eğitir.
test seti üzerinde modelin performansını değerlendirir ve metrikleri kaydeder.
"""

from __future__ import annotations # Python 3.10+ için tip ipuçlarını etkinleştirir
from pathlib import Path # dosya yolları ile çalışmayı kolaylaştırır
import json # metriklerin JSON formatında kaydedilmesi için
import joblib # modeli ve diğer nesneleri kaydetmek/yüklemek için
import pandas as pd 
from sklearn.linear_model import LinearRegression # Linear Regression modelini kullanmak için
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # performans metrikleri

from .config import(
    TRAIN_DATA, 
    TEST_DATA,
    TARGET,
    FEATURES_NUM,
    FEATURES_CAT,
    MODEL_PATH,
    METRICS_PATH
    )

def _check_required_columns(df: pd.DataFrame, name: str): # Gerekli sütunların varlığını kontrol eder
    required = FEATURES_NUM + FEATURES_CAT + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"[train] {name} Eksik sütunlar: {missing}")

def _coef_summary(model: LinearRegression, feature_names: list[str]) -> pd.DataFrame: # Model katsayılarını özetler yorumlayabilmek için
    coefs = pd.Series(model.coef_, index=feature_names, name="coefficient")
    return coefs.to_frame().sort_values(by="coefficient", ascending=False)

def main():
    # verilen eğitim ve test setlerini yükle
    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)

    # Gerekli sütunların varlığını kontrol et
    _check_required_columns(train_df, "Eğitim seti")
    _check_required_columns(test_df, "Test seti")

    # Özellikler ve hedef değişkeni ayır
    x_cols = FEATURES_NUM + FEATURES_CAT
    X_train , y_train = train_df[x_cols], train_df[TARGET]
    X_test , y_test = test_df[x_cols], test_df[TARGET]

    # Linear Regression modelini oluştur ve eğit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    # Model performans metriklerini kaydet
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Model katsayılarını özetle ve yazdır
    coef_df = _coef_summary(model, x_cols)
    print("\n[train] Model eğitimi tamamlandı.")
    print(f"[train] Model : {MODEL_PATH}")
    print(f"[train] Metrik : {METRICS_PATH} -> {metrics}")
    print(f"[train] Katsayı özeti (büyükten küçüğe)")
    print(coef_df.to_string(index=True))

if __name__ == "__main__":
    main()