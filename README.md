# Financial-Time-Series-Analysis
# Stratejik Sektörler Zaman Serisi Analizi ve Karar Destek Sistemi

Bu proje; Türkiye'nin lokomotif sektörlerini (Enerji, Savunma, Havacılık) temsil eden **TUPRS, ASELSAN ve THY** şirketlerinin 2015-2025 dönemi verilerini kullanarak ileri seviye istatistiksel modelleme ve görselleştirme sunar.

##  Projenin Amacı
Finansal zaman serilerindeki trend, mevsimsellik ve volatilite bileşenlerini ayrıştırarak, kurumsal olayların (temettü, bedelsiz sermaye artırımı) hisse fiyatları üzerindeki etkisini "Event Study" yaklaşımıyla analiz etmek ve interaktif bir karar destek mekanizması oluşturmak.

## Teknik Yetkinlikler & Metodoloji
- **İstatistiksel Testler:** Durağanlık analizi için **ADF (Augmented Dickey-Fuller)** testi, bağımlılık yapısı için **ACF/PACF** grafikleri.
- **Zaman Serisi Ayrıştırma:** Verinin; Trend, Mevsimsellik ve Artık (Residual) bileşenlerine **STL Decomposition** ile bölünmesi.
- **Risk ve Volatilite:** **Bollinger Bantları** ve **RSI** göstergeleri ile piyasa oynaklığının takibi.
- **Event Study:** Temettü ve bölünme tarihlerinin hisse getirileri üzerindeki etkisinin analizi.

##  Kullanılan Teknolojiler
- **Programlama:** Python (Pandas, NumPy, Statsmodels)
- **Görselleştirme:** Plotly (İnteraktif Grafikler)
- **Arayüz:** Streamlit (Dinamik Analiz Dashboard'u)

##  Dosya Yapısı
- `app.py`: Streamlit dashboard uygulama kodu.
- `analiz.ipynb`: Veri ön işleme ve istatistiksel analiz adımları.
- `requirements.txt`: Gerekli kütüphane listesi.
