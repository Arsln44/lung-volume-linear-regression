"""
Bu dosyada Kaggle'den veri seti indirme işlemi gerçekleştirilir.
"""

from pathlib import Path
import pandas as pd
from kagglehub import dataset_download
from .config import KAGGLE_DATASET, KAGGLE_FILE, RAW_DATA

def main():

    local_dir = Path(dataset_download(KAGGLE_DATASET)) # Kaggle veri setini indir ve yerel dizini al
    src_file = local_dir / KAGGLE_FILE # İndirilen dosyanın tam yolu

    if not src_file.exists(): # Dosyanın varlığını kontrol et
        available = [f.name for f in local_dir.glob("**\*.csv")] # Mevcut CSV dosyalarını listele
        raise FileNotFoundError(
            f"{KAGGLE_FILE} bulunamadı. Mevcut dosyalar: {available}" # Hata mesajını özelleştir
            )
    
    df = pd.read_csv(src_file) # CSV dosyasını oku
    df.to_csv(RAW_DATA, index=False) # Veriyi RAW_DATA yoluna koy

    print(f"[download] {len(df)} satır ve {len(df.columns)} sütun ile veri indirildi -> {RAW_DATA}")

if __name__ == "__main__":
    main()