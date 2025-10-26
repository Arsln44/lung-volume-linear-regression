# Linear Regression - Lung Volume Prediction

Basit ama profesyonel yapılandırılmış bir **doğrusal regresyon projesi**.  
Amaç, yaş (`age`), boy (`height`) ve cinsiyet (`sex`) verilerinden bir kişinin **akciğer kapasitesini (Litre cinsinden)** tahmin etmektir.

---

##  Projenin Amacı
Bu proje, veri bilimi öğrenme sürecinde:
- Gerçek bir veri setini **Kaggle API** üzerinden indirmeyi,  
- Veriyi **temizleyip ölçeklendirmeyi**,  
- Basit ama profesyonel bir **Linear Regression modeli** eğitmeyi,  
- Eğitim, test ve tahmin aşamalarını **modüler bir pipeline** üzerinden yürütmeyi öğretir.

---

##  Kullanılan Teknolojiler ve Kütüphaneler

| Kütüphane | Amaç |
|------------|------|
| **pandas** | Veri okuma, işleme, DataFrame yönetimi |
| **numpy** | Sayısal işlemler |
| **scikit-learn** | Linear Regression, train/test split, scaler |
| **matplotlib** | Görselleştirme |
| **joblib** | Model ve scaler dosyalarını kaydetme/yükleme |
| **kaggle / kagglehub** | Kaggle veri setini otomatik indirme |
| **tqdm** | İlerleme göstergesi |
| **jupyter** | Notebook ortamında analiz |

---

##  Çalışma Aşamaları (Pipeline)

1. **Veri İndirme (`src/download.py`)**
   - Kaggle API üzerinden veri çekilir.
   - `data/raw/data.csv` olarak kaydedilir.

2. **Veri Hazırlama (`src/prepare.py`)**
   - Gereksiz kolonlar (`Unnamed: 0`) temizlenir.
   - `sex` kolonu `M → 1`, `F → 0` olarak dönüştürülür.
   - Sayısal kolonlar (`age`, `height`) ölçeklenir (`StandardScaler`).
   - Veriler `train.csv` ve `test.csv` olarak kaydedilir.

3. **Model Eğitimi (`src/train.py`)**
   - Linear Regression modeli eğitilir.
   - Katsayılar (`coef`) ve metrikler (`MAE`, `MSE`, `R²`) raporlanır.
   - Model `models/lr_model.pkl` olarak kaydedilir.

4. **Tahmin (`src/predict.py`)**
   - Model ve scaler yüklenir.
   - Tekil veya toplu tahminler yapılır (`predict_dict`, `batch_predict`).

5. **Görselleştirme (`scripts/visualize.py`)**
   - Histogram, korelasyon ısı haritası, gerçek-tahmin karşılaştırması grafikleri.

---


# Dikkat Edilmesi Gerekenler

**kaggle.json** dosyanın konumu doğru olmalı (~/.kaggle/ veya proje dizini).

**src/config.py** içindeki **KAGGLE_DATASET** kimliği güncel olmalı.

**prepare.py** sırasında scaler oluşturulur; bu scaler **tahmin aşamasında mutlaka kullanılır.**

**Notebook’taki** hücreleri sırasıyla çalıştır — pipeline bir defa oluşturulunca model dosyaları tekrar kullanılabilir.

Yeni veriyle tahmin yapılırken kolon isimleri **birebir aynı olmalı** (age, height, sex).


# Model Sonuçları

| Metric  | Value  |
| ------- | ------ |
| **MAE** | 0.1964 |
| **MSE** | 0.0554 |
| **R²**  | 0.924  |


# Öğrenilen Konular

-> Modüler Python proje yapısı

-> Veri ölçeklendirme (StandardScaler)

-> Linear Regression katsayı analizi

-> Model ve scaler kaydetme/yükleme

-> Pipeline mantığı

-> Görselleştirme ve EDA


