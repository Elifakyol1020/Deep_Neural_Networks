# Ödev-1 - CIFAR-10 ile k-NN Sınıflandırma

Bu projede CIFAR-10 veri seti üzerinde k-NN (k-Nearest Neighbors) algoritması uygulanmıştır.

## Özellikler
- Veri seti yerel klasörden okunur.
- Kullanıcı L1 (Manhattan) veya L2 (Öklid) uzaklık metriğini seçer.
- Kullanıcı k değeri girer.
- Kullanıcı bir test örneği seçer.
- Seçilen örnek sınıflandırılır.
- Sonuç ve en yakın komşular ekranda gösterilir.

## Klasör Yapısı

├── data/
│   └── cifar-10/cifar-10/
├── main.py
├── requirements.txt
└── README.md

## Kurulum

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Çalıştırma

python main.py