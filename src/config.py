"""
bu dosya yapılandırma ayarlarını içerir.
proje boyunca kullanılan sabitler ve parametreler burada tanımlanır.
"""

from pathlib import Path

# Proje kök dizini
# Bu değişken, proje içindeki diğer dosyalara dinamik olarak erişmek için kullanılabilir.
ROOT = Path(__file__).resolve().parent.parents[0]

"""
Veri ayarlamaları
Bu bölümde veri ile ilgili sabitler ve parametreler tanımlanır.
"""

# Tahmin edilecek hedef değişkenin adı
TARGET = "IC"

# Sayısal özelliklerin listesi (Ölçeklendirme için)
FEATURES_NUM = ["age","height"]

# Kategorik özelliklerin listesi
FEATURES_CAT = ["sex"]

# Train/Test oranı (örneğin, %80 eğitim, %20 test)
TEST_SIZE = 0.2
# Rastgele durum sabiti (tekrarlanabilirlik için)
RANDOM_STATE = 42

# Ölçeklendirme yöntemi
SCALER = "standard"


"""
Veri yolları
"""
RAW_DATA = ROOT / "data" / "raw" / "data.csv"
TRAIN_DATA = ROOT / "data" / "processed" / "train.csv"
TEST_DATA = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "lr_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
METRICS_PATH = ROOT / "reports" / "metrics.json"

"""
Script yolları
"""
SCRIPTS_DIR = ROOT / "scripts"

SCRIPTS = {
    "pipeline":   SCRIPTS_DIR / "pipeline.py",
    "visualize":  SCRIPTS_DIR / "visualize.py",
    "explore":    SCRIPTS_DIR / "explore.py",
    "predict":    SCRIPTS_DIR / "predict_tools.py",
    "__init__":   SCRIPTS_DIR / "__init__.py"
}


""""
Kaggle ayarları
Bu bölümde Kaggle ile ilgili yapılandırma ayarları tanımlanır.
"""

KAGGLE_DATASET = "djathidiro/gli-lung-vol"
KAGGLE_FILE = "dataset.csv"