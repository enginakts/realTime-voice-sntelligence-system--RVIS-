# TIMIT Üzerinde Gerçek Zamanlı Cinsiyet Tahmini (CNN + BiLSTM)

Bu proje, **TIMIT** konuşma veri setini kullanarak **cinsiyet sınıflandırması (Male/Female)** yapan bir modelin eğitilmesini ve iki farklı “gerçek zamanlı” senaryoda test edilmesini amaçlar.

Genel fikir:

- TIMIT’ten ses yolları ve anotasyonları tek bir tabloya (CSV) toplanır.
- 16 kHz’e sabitlenmiş ses üzerinden **log-mel** özellikleri çıkarılır.
- CNN + BiLSTM tabanlı bir model eğitilir.
- Model, kullanıcı arayüzü üzerinden hem **canlı mikrofon** hem de **birleştirilmiş ses dosyası** senaryolarında pencere bazlı olarak çalıştırılır.

---

## Klasör Yapısı

- `data/`
  - `lisa/data/timit/raw/TIMIT/`  
    TIMIT’in orijinal dizin yapısı (TRAIN/TEST, DR1–DR8, speaker klasörleri, WAV/TXT/PHN/WRD).

- `code/`
  - `analiz.py`  
    Veri hazırlama + model eğitimi + test raporları. Dosya **`# %%` hücreleri** ile yazıldığı için Jupyter/VS Code Interactive ile hücre hücre çalıştırılabilir.
  - `csv_out/`
    - `timit_train.csv`, `timit_test.csv` (üretilen çıktılar)

- `realtime_gender_ui/`
  - `gradio_app.py`  
    Gradio tabanlı arayüz (Senaryo 1: mikrofon, Senaryo 2: dosya birleştir+trim+analiz).
  - `app.py`  
    Streamlit tabanlı arayüz (webrtc ile mikrofon).
  - `audio_features.py`  
    Log-mel feature + sessizlik kesme + pencereleme yardımcıları.
  - `gender_infer.py`  
    Tek pencere için “sessizlik kontrolü + 3 saniyeye pad/crop + model tahmini”.
  - `model_gender_cnn_bilstm.py`  
    CNN+BiLSTM mimarisi ve model yükleyici.
  - `requirements.txt`  
    UI tarafı için bağımlılıklar (zamanla sadeleştirilebilir).

---

## Veri Seti (TIMIT) Hakkında Kısa Not

- **Cinsiyet etiketi** `speaker_id`’nin ilk harfinden türetilir:
  - `M...` → `gender=0` (Male)
  - `F...` → `gender=1` (Female)
- `PHN` / `WRD` segment dosyalarındaki `start/end` değerleri **örnek indeksidir** ve genelde **16 kHz** ile uyumludur.

Not: TIMIT’te genelde **M sayısı F’den fazla** (sende yaklaşık %70 / %30). Bu yüzden sadece accuracy değil **macro-F1** ve **balanced accuracy** gibi metriklere de bakılıyor.

---

## Eğitim (Modeli Üretme)

Eğitim kodu: `code/analiz.py`

Bu dosya şunları yapar:

1. TIMIT’i tarar, her utterance için `.WAV/.TXT/.PHN/.WRD` yollarını ve segmentleri toplar.
2. `gender` etiketini `speaker_id` üzerinden üretir.
3. İsteğe bağlı olarak `PHN` ile baş/son sessizliği kırpar (dataset içinde “ideal VAD” gibi).
4. Log-mel çıkarır ve 3 saniyeye sabitler:
   - Train: random crop
   - Val/Test: center crop
5. Train/Val ayrımını **konuşmacıya göre** yapar (speaker-disjoint) — aynı konuşmacının hem train hem val’de olması val skorunu şişirebilir.
6. Dengesizlik için:
   - `SpeakerBalancedSampler` (varsayılan) veya `WeightedRandomSampler`
   - Balanced class weights ile `CrossEntropyLoss`
7. Checkpoint’i kaydeder:
   - `code/cnn_bilstm_gender_best.pt`

> Model mimarisi UI tarafındaki `realtime_gender_ui/model_gender_cnn_bilstm.py` ile uyumludur.

---

## Arayüz: Senaryo 1 ve Senaryo 2

### Senaryo 1 — Canlı Mikrofon ile Sürekli Tahmin

Amaç: Modelin **düşük gecikmeyle canlı ortamda** çalışıp çalışmadığını görmek.

Yaklaşım:

- Mikrofon akışı 16 kHz’e getirilir.
- Kısa çerçevelerde RMS(dB) ile **sessizlik filtrelenir**.
- Konuşma sayılan örneklerden kaydırmalı pencereler alınır.
- Her pencere için tahmin üretilir ve ekrana yazılır.

Gradio sürümünde Start’a basınca:

- Start butonu gizlenir,
- Stop ve “Recording…” animasyonu görünür,
- Son ~2 saniyenin dalga formu çizilir.

### Senaryo 2 — Birleştirilmiş Ses Dosyası ile “Gerçek Zamanlı” Analiz

Amaç: Dataset dışı ve konuşmacı geçişli bir akışta modelin tepkisini ölçmek.

- 10 kadın + 10 erkek (dataset dışı da olabilir) ses dosyası seçilir.
- Dosyalar uç uca eklenir; araya örn. **5 sn sessizlik** konur.
- Tek bir akış “stream ediliyormuş gibi” kaydırmalı pencerelerle analiz edilir.
- Trim sonrası pencere sonuçları tablo + olasılık grafiği olarak gösterilir.

---

## Çalıştırma (Gradio)

Gradio sürümü: `realtime_gender_ui/gradio_app.py`

```bash
cd "c:\Users\engin\Desktop\TIMIT\realtime_gender_ui"
pip install -r requirements.txt
python gradio_app.py
```

Terminalde `Running on local URL: http://127.0.0.1:....` linki çıkar.

---

## Çalıştırma (Streamlit)

Streamlit sürümü: `realtime_gender_ui/app.py`

```bash
cd "c:\Users\engin\Desktop\TIMIT\realtime_gender_ui"
pip install -r requirements.txt
streamlit run app.py
```

Not: Mikrofon için `streamlit-webrtc` ve bazı ortamlarda `av/aiortc` gerekebilir.

---

## Model Dosyası

Eğitim çıktısı varsayılan olarak:

- `code/cnn_bilstm_gender_best.pt`

UI tarafında model yolu, arayüzdeki “Model path” alanından değiştirilebilir.

---

## Sık Karşılaşılan Sorunlar

### 1) `OSError: WinError 127 ... fbgemm.dll`

Bu hata genelde Windows’ta PyTorch DLL bağımlılıkları ile ilgilidir.

Öneriler:

- **VC++ Redistributable 2015–2022 (x64)** kurulu değilse kur.
- Aynı conda/env içinde torch’u temizleyip tekrar kur:

```bash
conda activate <env>
pip uninstall -y torch torchvision torchaudio
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```

### 2) Mikrofon “başladı mı” belli değil

Gradio sürümünde Start sonrası “Recording…” animasyonu ve dalga formu var. Sessiz ortamda dalga küçük görünüyorsa “Sessizlik eşiği (RMS dB)” değerini biraz düşürmek işe yarar (örn. `-55`).

---

## Notlar / Geliştirme Fikirleri

- `realtime_gender_ui/requirements.txt` zamanla ikiye ayrılabilir:
  - `requirements_gradio.txt`
  - `requirements_streamlit.txt`
- Senaryo 2 için konuşmacı değişimlerini otomatik tespit eden bir “change-point” analizi eklenebilir.

