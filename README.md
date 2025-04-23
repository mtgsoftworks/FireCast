Orman Yangınları Veri Seti İstatistiksel Analizi
Proje Açıklaması
Bu proje, ISE-216 Veri Bilimi İçin İstatistik dersinin dönem projesi olarak geliştirilmiştir. Projenin amacı, orman yangınları veri setini kullanarak istatistiksel analizler yapmak ve bu veri seti üzerinde ders kapsamında öğrenilen teknikleri uygulamaktır.
Veri Seti Hakkında
Bu projede kullanılan veri seti, UCI Machine Learning Repository'den alınan "Forest Fires" veri setidir. Bu veri seti, Portekiz'in kuzeydoğusundaki Montesinho Doğal Parkı'ndaki orman yangınlarına ilişkin bilgileri içermektedir. Veri seti, toplam 517 örnek ve 13 özellik içermektedir.
Özellikler:

X: x-eksenindeki konumsal koordinat
Y: y-eksenindeki konumsal koordinat
month: yangının gerçekleştiği ay (Ocak-Aralık)
day: yangının gerçekleştiği gün (Pazartesi-Pazar)
FFMC: Ince Yakıt Nem Kodu (Fine Fuel Moisture Code) - yangın tehlikesi göstergesi
DMC: Duff Moisture Code - yangın tehlikesi göstergesi
DC: Drought Code - yangın tehlikesi göstergesi
ISI: Initial Spread Index - yangının yayılma hızı endeksi
temp: sıcaklık (°C)
RH: bağıl nem (%)
wind: rüzgar hızı (km/h)
rain: yağmur (mm/m²)
area: yanmış alan (hektar)

Proje Hedefleri

Veri setini yüklemek ve incelemek
Veri temizleme ve ön işleme yapmak
Keşifsel veri analizi gerçekleştirmek
Tanımlayıcı istatistikler çıkarmak
Hipotez testleri uygulamak
Korelasyon analizi yapmak
Değişkenler arasındaki ilişkileri görselleştirmek
İstatistiksel modeller oluşturmak
Sonuçları yorumlamak ve raporlamak

Gerekli Kütüphaneler

pandas: Veri manipülasyonu ve analizi
numpy: Bilimsel hesaplama
matplotlib: Veri görselleştirme
seaborn: İstatistiksel veri görselleştirme
scipy: Bilimsel ve teknik hesaplama
statsmodels: İstatistiksel modelleme ve hipotez testleri

Proje İçeriği

Veri Yükleme ve İnceleme
Veri Temizleme ve Ön İşleme
Keşifsel Veri Analizi
Tanımlayıcı İstatistikler
Korelasyon Analizi
Hipotez Testleri

Normallik Testleri
t-Testleri
ANOVA
Ki-Kare Testi


Görselleştirmeler
İstatistiksel Modelleme
Sonuçlar ve Değerlendirme

Nasıl Çalıştırılır

Gerekli kütüphaneleri yükleyin:

bashpip install pandas numpy matplotlib seaborn scipy statsmodels

forestfires.py dosyasını çalıştırın:

bashpython forestfires.py

Analiz sonuçları ve görselleştirmeler ekrana yazdırılacak ve /outputs klasörüne kaydedilecektir.

Katkıda Bulunanlar

Mesut Taha Güven

Kaynaklar

UCI Machine Learning Repository - Forest Fires Data Set
https://archive.ics.uci.edu/dataset/162/forest+fires
https://www.kaggle.com/datasets/elikplim/forest-fires-data-set/data