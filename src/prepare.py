"""
Bu dosyada indirdiğimiz ham veriyi işler
gereksiz sütunları kaldırır ve eğitim/test setlerine böleriz.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from .config import( 
    RAW_DATA, 
    TRAIN_DATA, 
    TEST_DATA,
    TARGET,
    FEATURES_NUM,
    FEATURES_CAT,
    TEST_SIZE,
    RANDOM_STATE,
    SCALER,
    SCALER_PATH
    )

def _make_scaler(kind: str | None): # Config dosyasındaki SCALER ayarına göre uygun ölçekleyiciyi döndürür
    if kind == "standard":
        return StandardScaler()
    elif kind == "minmax":
        return MinMaxScaler()
    else:
        return None
    
def main():
    # Ham veriyi oku
    df = pd.read_csv(RAW_DATA)

    # Gereksiz sütunları kaldır (örneğin hedef değişken ve özellikler dışındaki sütunlar)
    drop_col = [col for col in df.columns if col.lower().startswith("unnamed")]
    if drop_col:
        df = df.drop(columns=drop_col)
        print(f"[prepare] Gereksiz sütunlar kaldırıldı: {drop_col}")

    # Gerkekli sütunların varlığını kontrol et
    required = FEATURES_NUM + FEATURES_CAT + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"[prepare] Eksik sütunlar: {missing}")

    # Eksizlik verileri kaldır
    before = len(df)
    df.dropna(subset=required, inplace=True)
    print(f"[prepare] Eksik veriler kaldırıldı: {before - len(df)} satır silindi")

    # Kategorik değişkenleri 1 ve 0'a dönüştür (örneğin, erkek/kadın -> 1/0)
    for col in FEATURES_CAT:
        df[col] = df[col].map({'M' : 1, 'F' : 0})

    # Numerik özelliklerin türünü kontrol et
    for col in FEATURES_NUM:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Train/Test setlerine böl
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True, # Veriyi karıştırarak böl
    )

    # Sayısal özellikleri ölçeklendir
    scaler = _make_scaler(SCALER)
    if scaler and FEATURES_NUM:
        train_df[FEATURES_NUM] = scaler.fit_transform(train_df[FEATURES_NUM])
        test_df[FEATURES_NUM] = scaler.transform(test_df[FEATURES_NUM])
        print(f"[prepare] Sayısal özellikler '{SCALER}' ile ölçeklendirildi: {FEATURES_NUM}")

        # Ölçekleyiciyi kaydet
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[prepare] Ölçekleyici kaydedildi: {SCALER_PATH}")

    # İşlenmiş veriyi kaydet
    TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    TEST_DATA.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA, index=False)
    test_df.to_csv(TEST_DATA, index=False)

    print(f"[prepare] Eğitim seti kaydedildi: {len(train_df)} satır -> {TRAIN_DATA}")
    print(f"[prepare] Test seti kaydedildi: {len(test_df)} satır -> {TEST_DATA}")

if __name__ == "__main__":
    main()