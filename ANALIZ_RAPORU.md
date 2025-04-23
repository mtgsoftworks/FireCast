Orman Yangınları Veri Seti Analiz Raporu
Özet
Bu rapor, ISE-216 Veri Bilimi İçin İstatistik dersi kapsamında gerçekleştirilen orman yangınları veri seti analiz çalışmasının sonuçlarını içermektedir. Çalışmada, 517 satır ve 13 değişkenden oluşan Forest Fires veri seti üzerinde kapsamlı bir istatistiksel analiz uygulanmıştır. Veri temizleme, keşifsel veri analizi, tanımlayıcı istatistikler, korelasyon analizi, hipotez testleri ve istatistiksel modelleme gibi adımlar gerçekleştirilmiştir. Bu rapor, çalışmanın bulgularını ve sonuçlarını detaylı bir şekilde sunmaktadır.
1. Veri Seti Hakkında
Forest Fires veri seti, Portekiz'in Montesinho Doğal Parkı bölgesinde meydana gelen orman yangınlarına ilişkin verileri içermektedir. Veri seti, 517 satır ve 13 değişkenden oluşmaktadır. Değişkenler arasında konumsal koordinatlar, meteorolojik faktörler, yangın tehlikesi indeksleri ve yanmış alan bilgileri bulunmaktadır.
Veri seti başlangıçta temiz olduğu için, proje kapsamında veriye kasıtlı olarak eksik değerler ve aykırı değerler eklenerek veri temizleme ve ön işleme adımlarının uygulanması sağlanmıştır.
2. Veri Ön İşleme ve Temizleme
Eksik Değerler

Veri setine rastgele konumlarda 20 adet eksik değer eklenmiştir.
Sayısal değişkenler için eksik değerler medyan ile, kategorik değişkenler için mod ile doldurulmuştur.

Aykırı Değerler

Sayısal değişkenlere aykırı değerler eklenmiştir.
Çeyrekler arası aralık (IQR) yöntemi kullanılarak aykırı değerler tespit edilmiştir.
Aykırı değerler, clipping yöntemi ile alt ve üst sınırlar arasına alınarak düzeltilmiştir.

Veri Dönüşümü

Yanmış alan (area) değişkeni için logaritmik dönüşüm uygulanmıştır (log_area = log(area + 1)).
Ay ve gün değişkenleri sıralı kategorik değişkenler olarak yeniden düzenlenmiştir.
Mevsimsel analiz için ay değişkeninden season (mevsim) değişkeni türetilmiştir.
Haftaiçi/haftasonu analizi için is_weekend değişkeni oluşturulmuştur.

3. Keşifsel Veri Analizi Bulguları
Kategorik Değişkenlerin Dağılımları

Yangınların aylara göre dağılımı incelendiğinde, yaz aylarında (özellikle Ağustos ve Eylül) daha fazla yangın meydana geldiği görülmüştür.
Haftanın günlerine göre dağılımda belirgin bir farklılık gözlenmemiştir.

Sayısal Değişkenlerin Dağılımları

Yanmış alan (area) değişkeni oldukça sağa çarpık bir dağılım göstermektedir; çoğu yangın küçük alanları etkilemektedir.
Logaritmik dönüşüm sonrası alan dağılımı daha normal bir dağılıma yaklaşmıştır.
Yangın tehlikesi göstergeleri (FFMC, DMC, DC, ISI) genellikle orta-yüksek değerlerde yoğunlaşmaktadır.

Konumsal Dağılım

Park içerisindeki bazı bölgelerde (örneğin, belirli X-Y koordinatlarında) daha şiddetli yangınların meydana geldiği tespit edilmiştir.
Bu durum, coğrafi faktörlerin yangın şiddeti üzerinde etkisi olduğunu göstermektedir.

4. Tanımlayıcı İstatistikler
Mevsimsel Analiz

Yangınların mevsimsel dağılımında, yaz aylarında (Haziran-Ağustos) en yüksek yangın sayısı kaydedilmiştir (toplam yangınların yaklaşık %40'ı).
Ortalama yanmış alan büyüklüğü de yaz mevsiminde diğer mevsimlere göre daha yüksektir.

Hafta İçi vs Hafta Sonu

Haftaiçi günlerde 370 yangın, haftasonu günlerde 147 yangın kaydedilmiştir.
Ortalama yanmış alan büyüklüğü açısından haftaiçi ve haftasonu arasında önemli bir fark gözlenmemiştir.

5. Korelasyon Analizi
Değişkenler Arası İlişkiler

En güçlü pozitif korelasyon DMC ve DC arasında gözlemlenmiştir (r ≈ 0.70).
ISI ile FFMC arasında orta düzeyde pozitif korelasyon bulunmaktadır (r ≈ 0.53).
Sıcaklık (temp) ve bağıl nem (RH) arasında negatif korelasyon vardır (r ≈ -0.45).
Yanmış alan (area) ile diğer değişkenler arasındaki korelasyonlar genellikle zayıftır.

6. Hipotez Testleri
Normallik Testleri

Shapiro-Wilk testi sonuçlarına göre, çoğu değişken normal dağılım göstermemektedir (p < 0.05).
Log dönüşümü yapılmış alan değişkeni (log_area) de normal dağılım göstermemektedir.

ANOVA Testi

Mevsimler arasında yanmış alan (log_area) açısından istatistiksel olarak anlamlı farklılık bulunmuştur (F = 4.27, p = 0.0054).
Tukey HSD Post-Hoc testi sonuçlarına göre, özellikle Yaz-Kış ve Yaz-İlkbahar mevsimleri arasında anlamlı farklılıklar tespit edilmiştir.

t-Test

Haftaiçi ve haftasonu yangın şiddeti arasında istatistiksel olarak anlamlı bir fark bulunmamıştır (t = -0.85, p = 0.3942).

Ki-Kare Bağımsızlık Testi

Ay ve gün arasındaki ilişki incelendiğinde, bu iki kategorik değişken arasında istatistiksel olarak anlamlı bir ilişki bulunmamıştır (χ² = 68.94, p = 0.2143).

7. İstatistiksel Modelleme
Çoklu Regresyon Analizi

Yangın alanını (log_area) tahmin etmek için FFMC, DMC, DC, ISI, temp, RH, wind ve rain değişkenleri kullanılarak çoklu regresyon modeli oluşturulmuştur.
Model sonuçlarına göre:

R² = 0.0412 (Düzeltilmiş R² = 0.0263)
İstatistiksel olarak anlamlı değişkenler: temp (p = 0.0345), DC (p = 0.0289)
Diğer değişkenler istatistiksel olarak anlamlı bulunmamıştır (p > 0.05)


Model, yangın alanı varyansının yalnızca yaklaşık %4'ünü açıklayabilmektedir, bu da yangın davranışını etkileyen başka önemli faktörlerin olduğunu göstermektedir.

8. Sonuçlar ve Yorumlar

Mevsimsel Etkiler: Yaz aylarında hem yangın sayısı hem de yangın şiddeti artış göstermektedir. Bu durum, yüksek sıcaklık ve düşük nem gibi iklim faktörlerinin yangın riskini artırdığını doğrulamaktadır.
Yangın İndeksleri: FFMC, DMC, DC ve ISI gibi yangın tehlikesi indeksleri yangın alanı ile pozitif ilişki göstermektedir, ancak bu ilişkiler beklenenin altında kalmıştır.
Meteorolojik Faktörler: Sıcaklık ve nem, yangın davranışını etkileyen önemli faktörlerdir. Sıcaklık artışı ve nem düşüşü, yangın riskini artırmaktadır.
Tahmin Zorluğu: Oluşturulan regresyon modelinin düşük açıklayıcı gücü, orman yangınlarının tahmin edilmesinin zorluğunu göstermektedir. Yangın davranışı, ölçülmeyen veya veri setine dahil edilmeyen birçok faktörden etkileniyor olabilir.
Veri Kalitesi: Yanmış alan verisinin yüksek derecede çarpık olması, istatistiksel analizlerde zorluklar yaratmaktadır. Logaritmik dönüşüm bu sorunu kısmen hafifletse de, yangın alanının tahmin edilmesi hala zor bir problem olarak kalmaktadır.

9. Öneriler

Veri Zenginleştirme: Gelecekteki çalışmalarda, bitki örtüsü tipi, arazi eğimi, toprak nemi gibi ek faktörlerin dahil edilmesi modelin açıklayıcı gücünü artırabilir.
İleri Modelleme Teknikleri: Doğrusal regresyon yerine, doğrusal olmayan ilişkileri daha iyi yakalayabilen makine öğrenimi yöntemleri (örneğin, rastgele orman, gradyan artırma) kullanılabilir.
Zamansal Analiz: Yangın olayları arasındaki zamansal bağımlılıkları incelemek için zaman serisi analizi yapılabilir.
Mekânsal Analiz: Yangın konumları arasındaki mekânsal bağımlılıkları incelemek için jeo-istatistiksel yöntemler uygulanabilir.
Risk Haritalaması: Elde edilen bulgulara dayanarak, yangın risk bölgelerinin belirlenmesi ve haritalandırılması yangın yönetimi stratejilerine katkı sağlayabilir.

Bu analiz, orman yangınlarının karmaşık doğasını anlamada ve yangın davranışını etkileyen faktörleri belirlemede önemli bilgiler sağlamaktadır. Ancak, yangın davranışının tam olarak tahmin edilebilmesi için daha kapsamlı veri setleri ve gelişmiş modelleme teknikleri gerektiği açıktır.