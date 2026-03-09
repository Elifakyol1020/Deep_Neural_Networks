# Ödev-1 - CIFAR-10 ile k-NN Sınıflandırma

Bu projede CIFAR-10 veri seti üzerinde k-NN (k-Nearest Neighbors) algoritması uygulanmıştır.

## Özellikler
- Veri seti yerel klasörden okunur.
- Web arayüzünde kullanıcı L1 (Manhattan) veya L2 (Öklid) uzaklık metriğini seçer.
- Kullanıcı `k` ve sınıf başına kullanılacak örnek sayısını girer.
- Kullanıcı test görsel yolunu girer.
- Tahmin edilen sınıf, gerçek sınıf ve test görseli birlikte gösterilir.

## Klasör Yapısı

├── data/
│   └── cifar10/cifar10/
├── app.py
├── main.py
├── requirements.txt
└── README.md

## Kurulum

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Çalıştırma

Web arayüzü:

streamlit run app.py

Komut satırı sürümü:

python main.py
