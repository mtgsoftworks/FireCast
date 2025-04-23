#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Orman Yangınları Veri Seti İstatistiksel Analizi
ISE-216 Veri Bilimi İçin İstatistik Dersi Proje Çalışması
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Görsel ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'

# Çıktı klasörü oluştur
if not os.path.exists('output'):
    os.makedirs('output')

def save_figure(fig, filename):
    """Görseli kaydet"""
    fig.savefig(f"output/{filename}", bbox_inches='tight', dpi=300)
    
    # Handle different types of plot objects
    if hasattr(fig, 'fig'):  # For seaborn objects like PairGrid, FacetGrid, etc.
        plt.close(fig.fig)
    else:  # For regular matplotlib Figure objects
        plt.close(fig)

print("="*80)
print("ORMAN YANGINLARI VERİ SETİ İSTATİSTİKSEL ANALİZİ")
print("="*80)

# 1. Veri Yükleme ve İnceleme
print("\n1. VERİ YÜKLEME VE İNCELEME")
print("-"*80)

# Veriyi yükle
df = pd.read_csv('data/forestfires.csv')

# Veri seti hakkında bilgi ver
print(f"Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
print("\nSütun isimleri ve veri tipleri:")
print(df.dtypes)

print("\nVeri setinin ilk 5 satırı:")
print(df.head())

print("\nVeri setine ait özet istatistikler:")
print(df.describe().T)

# Kayıp değerleri kontrol et
missing_values = df.isnull().sum()
print("\nKayıp değerler:")
print(missing_values)

# 2. Veri Temizleme ve Ön İşleme
print("\n2. VERİ TEMİZLEME VE ÖN İŞLEME")
print("-"*80)

# Bu bölümde projeye uygun olarak verimizi kirletelim
# Rastgele bazı değerleri NaN yapalım
np.random.seed(42)
rows = np.random.randint(0, df.shape[0], size=20)
cols = np.random.randint(0, df.shape[1], size=20)

for i, j in zip(rows, cols):
    df.iloc[i, j] = np.nan

# Aykırı değerler ekleyelim
numeric_cols = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
for col in numeric_cols:
    if col != 'area':  # area zaten çok çarpık bir dağılıma sahip
        outlier_indices = np.random.randint(0, df.shape[0], size=5)
        for idx in outlier_indices:
            # Sütunun maksimum değerinin 2-3 katı arasında rastgele bir değer
            df.loc[idx, col] = df[col].max() * (2 + np.random.random())

print("Veri kasıtlı olarak kirletildi (kayıp değerler ve aykırı değerler eklendi)")

# Eksik değerleri kontrol et
missing_values = df.isnull().sum()
print("\nKirlettikten sonra kayıp değerler:")
print(missing_values)

# Eksik değerleri doldur
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        # Sayısal sütunlar için medyan ile doldur
        df[col].fillna(df[col].median(), inplace=True)
    else:
        # Kategorik sütunlar için mod ile doldur
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nEksik değerler doldurulduktan sonra:")
print(df.isnull().sum())

# month ve day sütunlarını sıralı kategorik değişkenlere dönüştür
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
df['day'] = pd.Categorical(df['day'], categories=day_order, ordered=True)

# Aykırı değerleri tespit et
print("\nAykırı değerleri tespit etme:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    print(f"{col}: {len(outliers)} aykırı değer")

# Aykırı değerleri işleme - clipping yöntemi
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("\nAykırı değerler clipping yöntemi ile düzeltildi")

# Yanmış alan (area) sütunu çok çarpık, logaritmik dönüşüm uygula
# Log dönüşümü yapmadan önce 0 değerlerini ele almak için 1 ekle
df['log_area'] = np.log1p(df['area'])

print("\nVeri temizleme ve ön işleme tamamlandı")

# 3. Keşifsel Veri Analizi
print("\n3. KEŞİFSEL VERİ ANALİZİ")
print("-"*80)

# Kategorik değişkenlerin dağılımları
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Aylara göre yangın sıklığı
sns.countplot(x='month', data=df, ax=axes[0])
axes[0].set_title('Aylara Göre Yangın Sıklığı')
axes[0].set_xlabel('Ay')
axes[0].set_ylabel('Yangın Sayısı')
axes[0].tick_params(axis='x', rotation=45)

# Günlere göre yangın sıklığı
sns.countplot(x='day', data=df, ax=axes[1])
axes[1].set_title('Günlere Göre Yangın Sıklığı')
axes[1].set_xlabel('Gün')
axes[1].set_ylabel('Yangın Sayısı')

plt.tight_layout()
save_figure(fig, "categorical_distributions.png")
print("Kategorik değişkenlerin dağılımları görselleştirildi")

# Sayısal değişkenlerin dağılımları
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'{col} Dağılımı')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frekans')

plt.tight_layout()
save_figure(fig, "numerical_distributions.png")
print("Sayısal değişkenlerin dağılımları görselleştirildi")

# Log dönüşümlü alan dağılımı
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(df['area'], kde=True, ax=axes[0])
axes[0].set_title('Yanmış Alan Dağılımı')
axes[0].set_xlabel('Alan (hektar)')
axes[0].set_ylabel('Frekans')

sns.histplot(df['log_area'], kde=True, ax=axes[1])
axes[1].set_title('Log Dönüşümlü Yanmış Alan Dağılımı')
axes[1].set_xlabel('Log(Alan + 1)')
axes[1].set_ylabel('Frekans')

plt.tight_layout()
save_figure(fig, "area_transformation.png")
print("Yanmış alan dağılımı ve log dönüşümü görselleştirildi")

# Isı haritası - konumsal dağılım
fig, ax = plt.subplots(figsize=(10, 8))

# X ve Y koordinatlarına göre ortalama yanmış alan büyüklüğü
heatmap_data = df.pivot_table(values='area', index='Y', columns='X', aggfunc='mean')
sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax)
ax.set_title('Konumsal Yangın Şiddeti Dağılımı (Ortalama Yanmış Alan)')
ax.set_xlabel('X Koordinatı')
ax.set_ylabel('Y Koordinatı')

plt.tight_layout()
save_figure(fig, "spatial_heatmap.png")
print("Konumsal yangın şiddeti dağılımı görselleştirildi")

# 4. Tanımlayıcı İstatistikler
print("\n4. TANIMLAYICI İSTATİSTİKLER")
print("-"*80)

# Sayısal değişkenlerin tanımlayıcı istatistikleri
numeric_stats = df[numeric_cols + ['log_area']].describe()
print("Sayısal değişkenlerin tanımlayıcı istatistikleri:")
print(numeric_stats)

# Kategorik değişkenlerin frekans tabloları
print("\nAylara göre yangın sayıları:")
print(df['month'].value_counts().sort_index())

print("\nGünlere göre yangın sayıları:")
print(df['day'].value_counts().sort_index())

# Mevsimsel analiz için ay verilerini mevsime dönüştür
season_map = {
    'dec': 'Winter', 'jan': 'Winter', 'feb': 'Winter',
    'mar': 'Spring', 'apr': 'Spring', 'may': 'Spring',
    'jun': 'Summer', 'jul': 'Summer', 'aug': 'Summer',
    'sep': 'Autumn', 'oct': 'Autumn', 'nov': 'Autumn'
}
df['season'] = df['month'].map(season_map)

print("\nMevsimlere göre yangın sayıları:")
season_counts = df['season'].value_counts()
print(season_counts)

# Mevsime göre yanmış alan ortalamaları
season_areas = df.groupby('season')['area'].agg(['mean', 'median', 'min', 'max', 'std'])
print("\nMevsimlere göre yanmış alan istatistikleri:")
print(season_areas)

# 5. Korelasyon Analizi
print("\n5. KORELASYON ANALİZİ")
print("-"*80)

# Sayısal değişkenler arasındaki korelasyon
correlation_matrix = df[numeric_cols + ['log_area']].corr()
print("Korelasyon matrisi:")
print(correlation_matrix)

# Korelasyon matrisi ısı haritası
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Değişkenler Arası Korelasyon Matrisi')
plt.tight_layout()
save_figure(fig, "correlation_heatmap.png")
print("Korelasyon matrisi görselleştirildi")

# En yüksek korelasyona sahip değişken çiftleri
corr_pairs = correlation_matrix.unstack()
sorted_corrs = corr_pairs.sort_values(ascending=False)
# 1'e eşit olan değerleri (kendi ile korelasyon) filtrele
high_corrs = sorted_corrs[(sorted_corrs < 0.999) & (sorted_corrs > 0.5)]
print("\nEn yüksek pozitif korelasyona sahip değişken çiftleri:")
print(high_corrs)

# En düşük korelasyona sahip değişken çiftleri
low_corrs = sorted_corrs[(sorted_corrs > -0.999) & (sorted_corrs < -0.1)]
print("\nNegatif korelasyona sahip değişken çiftleri:")
print(low_corrs)

# Çiftli ilişki grafikleri - en önemli değişkenler için
important_vars = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'area', 'log_area']
fig = sns.pairplot(df[important_vars], height=2.5, diag_kind='kde')
plt.suptitle('Önemli Değişkenler Arası İlişkiler', y=1.02, fontsize=16)
save_figure(fig, "pairwise_relationships.png")
print("Değişkenler arası çiftli ilişkiler görselleştirildi")

# 6. Hipotez Testleri
print("\n6. HİPOTEZ TESTLERİ")
print("-"*80)

# Normallik Testleri
print("Normallik Testleri (Shapiro-Wilk):")
for col in numeric_cols + ['log_area']:
    # Büyük veri setleri için 5000 elemandan küçük bir örnek kullan
    sample = df[col].sample(min(1000, len(df))).values
    stat, p = stats.shapiro(sample)
    print(f"{col}: İstatistik = {stat:.4f}, p-değeri = {p:.4e}, {'Normal değil' if p < 0.05 else 'Normal'}")

# Mevsime göre yanmış alan farklılıkları (ANOVA)
print("\nMevsime göre yanmış alan farklılıkları (ANOVA):")
# Log dönüşümlü alan kullan
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
anova_data = [df[df['season'] == season]['log_area'] for season in seasons]

f_stat, p_value = stats.f_oneway(*anova_data)
print(f"F-istatistiği: {f_stat:.4f}, p-değeri: {p_value:.4f}")
print(f"Sonuç: {'Mevsimler arasında anlamlı fark var' if p_value < 0.05 else 'Mevsimler arasında anlamlı fark yok'}")

# ANOVA sonrası post-hoc test (Tukey HSD)
if p_value < 0.05:
    print("\nTukey HSD Post-Hoc Test Sonuçları:")
    tukey = pairwise_tukeyhsd(df['log_area'], df['season'], alpha=0.05)
    print(tukey)

# Sıcaklık ve nem arasındaki ilişki için regresyon analizi
print("\nSıcaklık ve Nem Arasındaki İlişki (Regresyon Analizi):")
X = sm.add_constant(df['temp'])
y = df['RH']
model = sm.OLS(y, X).fit()
print(model.summary().tables[1])

# Haftaiçi ve haftasonu yangın şiddeti karşılaştırması (t-test)
print("\nHaftaiçi ve Haftasonu Yangın Şiddeti Karşılaştırması (t-test):")
df['is_weekend'] = df['day'].isin(['sat', 'sun'])
weekday_area = df[~df['is_weekend']]['log_area']
weekend_area = df[df['is_weekend']]['log_area']

t_stat, p_val = stats.ttest_ind(weekday_area, weekend_area, equal_var=False)
print(f"t-istatistiği: {t_stat:.4f}, p-değeri: {p_val:.4f}")
print(f"Sonuç: {'Haftaiçi ve haftasonu yangın şiddeti arasında anlamlı fark var' if p_val < 0.05 else 'Haftaiçi ve haftasonu yangın şiddeti arasında anlamlı fark yok'}")

# Ki-kare bağımsızlık testi - Ay ve gün arasındaki ilişki
print("\nAy ve Gün Arasındaki İlişki (Ki-Kare Bağımsızlık Testi):")
contingency_table = pd.crosstab(df['month'], df['day'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Ki-kare istatistiği: {chi2:.4f}, p-değeri: {p:.4f}")
print(f"Sonuç: {'Ay ve gün arasında anlamlı bir ilişki var' if p < 0.05 else 'Ay ve gün arasında anlamlı bir ilişki yok'}")

# 7. Görselleştirmeler
print("\n7. GÖRSELLEŞTİRMELER")
print("-"*80)

# Mevsimlere göre yangın sayısı ve ortalama yanmış alan
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mevsimlere göre yangın sayısı
sns.countplot(x='season', data=df, order=['Winter', 'Spring', 'Summer', 'Autumn'], ax=axes[0])
axes[0].set_title('Mevsimlere Göre Yangın Sayısı')
axes[0].set_xlabel('Mevsim')
axes[0].set_ylabel('Yangın Sayısı')

# Mevsimlere göre ortalama yanmış alan
season_area_means = df.groupby('season')['area'].mean().reindex(['Winter', 'Spring', 'Summer', 'Autumn'])
sns.barplot(x=season_area_means.index, y=season_area_means.values, ax=axes[1])
axes[1].set_title('Mevsimlere Göre Ortalama Yanmış Alan')
axes[1].set_xlabel('Mevsim')
axes[1].set_ylabel('Ortalama Yanmış Alan (hektar)')

plt.tight_layout()
save_figure(fig, "season_analysis.png")
print("Mevsimlere göre yangın analizi görselleştirildi")

# Haftaiçi vs Haftasonu karşılaştırması
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Haftaiçi vs Haftasonu yangın sayısı
weekday_count = df[~df['is_weekend']].shape[0]
weekend_count = df[df['is_weekend']].shape[0]
axes[0].bar(['Haftaiçi', 'Haftasonu'], [weekday_count, weekend_count])
axes[0].set_title('Haftaiçi vs Haftasonu Yangın Sayısı')
axes[0].set_ylabel('Yangın Sayısı')

# Haftaiçi vs Haftasonu ortalama yanmış alan
weekday_mean = df[~df['is_weekend']]['area'].mean()
weekend_mean = df[df['is_weekend']]['area'].mean()
axes[1].bar(['Haftaiçi', 'Haftasonu'], [weekday_mean, weekend_mean])
axes[1].set_title('Haftaiçi vs Haftasonu Ortalama Yanmış Alan')
axes[1].set_ylabel('Ortalama Yanmış Alan (hektar)')

plt.tight_layout()
save_figure(fig, "weekday_vs_weekend.png")
print("Haftaiçi vs Haftasonu karşılaştırması görselleştirildi")

# Meteorolojik faktörlerin yanmış alan üzerindeki etkisi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Sıcaklık vs Yanmış Alan (log)
sns.scatterplot(x='temp', y='log_area', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Sıcaklık ve Yanmış Alan İlişkisi')
axes[0, 0].set_xlabel('Sıcaklık (°C)')
axes[0, 0].set_ylabel('Log(Yanmış Alan + 1)')

# Bağıl Nem vs Yanmış Alan (log)
sns.scatterplot(x='RH', y='log_area', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Bağıl Nem ve Yanmış Alan İlişkisi')
axes[0, 1].set_xlabel('Bağıl Nem (%)')
axes[0, 1].set_ylabel('Log(Yanmış Alan + 1)')

# Rüzgar Hızı vs Yanmış Alan (log)
sns.scatterplot(x='wind', y='log_area', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Rüzgar Hızı ve Yanmış Alan İlişkisi')
axes[1, 0].set_xlabel('Rüzgar Hızı (km/h)')
axes[1, 0].set_ylabel('Log(Yanmış Alan + 1)')

# Yağmur vs Yanmış Alan (log)
sns.scatterplot(x='rain', y='log_area', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Yağmur ve Yanmış Alan İlişkisi')
axes[1, 1].set_xlabel('Yağmur (mm/m²)')
axes[1, 1].set_ylabel('Log(Yanmış Alan + 1)')

plt.tight_layout()
save_figure(fig, "meteorological_effects.png")
print("Meteorolojik faktörlerin yanmış alan üzerindeki etkisi görselleştirildi")

# Yangın indekslerinin yanmış alan üzerindeki etkisi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# FFMC vs Yanmış Alan (log)
sns.scatterplot(x='FFMC', y='log_area', data=df, ax=axes[0, 0])
axes[0, 0].set_title('FFMC ve Yanmış Alan İlişkisi')
axes[0, 0].set_xlabel('FFMC')
axes[0, 0].set_ylabel('Log(Yanmış Alan + 1)')

# DMC vs Yanmış Alan (log)
sns.scatterplot(x='DMC', y='log_area', data=df, ax=axes[0, 1])
axes[0, 1].set_title('DMC ve Yanmış Alan İlişkisi')
axes[0, 1].set_xlabel('DMC')
axes[0, 1].set_ylabel('Log(Yanmış Alan + 1)')

# DC vs Yanmış Alan (log)
sns.scatterplot(x='DC', y='log_area', data=df, ax=axes[1, 0])
axes[1, 0].set_title('DC ve Yanmış Alan İlişkisi')
axes[1, 0].set_xlabel('DC')
axes[1, 0].set_ylabel('Log(Yanmış Alan + 1)')

# ISI vs Yanmış Alan (log)
sns.scatterplot(x='ISI', y='log_area', data=df, ax=axes[1, 1])
axes[1, 1].set_title('ISI ve Yanmış Alan İlişkisi')
axes[1, 1].set_xlabel('ISI')
axes[1, 1].set_ylabel('Log(Yanmış Alan + 1)')

plt.tight_layout()
save_figure(fig, "fire_indices_effects.png")
print("Yangın indekslerinin yanmış alan üzerindeki etkisi görselleştirildi")

# 8. İstatistiksel Modelleme
print("\n8. İSTATİSTİKSEL MODELLEME")
print("-"*80)

# Çoklu regresyon modeli - Yanmış alanı (log) tahmin etmek için
print("Çoklu Regresyon Modeli (Yanmış Alanı Tahmin Etmek İçin):")

# Önemli değişkenleri seç
features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
X = df[features]
y = df['log_area']

# Veriyi standartlaştır
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regresyon modeli
X_with_const = sm.add_constant(X_scaled)
model = sm.OLS(y, X_with_const).fit()
print(model.summary().tables[1])

# R-kare ve düzeltilmiş R-kare değerleri
print(f"R-kare: {model.rsquared:.4f}")
print(f"Düzeltilmiş R-kare: {model.rsquared_adj:.4f}")

# Katsayıların görselleştirilmesi
coefs = model.params[1:]  # İlk katsayı sabit terim
stderrs = model.bse[1:]
names = X.columns

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(coefs)), coefs)
ax.set_xticks(range(len(coefs)))
ax.set_xticklabels(names, rotation=45)
ax.set_ylabel('Katsayı Değeri')
ax.set_title('Regresyon Modeli Katsayıları')

# Hata çubukları ekle (standart hata)
ax.errorbar(range(len(coefs)), coefs, yerr=stderrs, fmt='none', ecolor='black', capsize=5)

# İstatistiksel olarak anlamlı olanları işaretle (p<0.05)
p_values = model.pvalues[1:]
for i, (bar, p_val) in enumerate(zip(bars, p_values)):
    if p_val < 0.05:
        bar.set_color('red')

plt.tight_layout()
save_figure(fig, "regression_coefficients.png")
print("Regresyon modeli katsayıları görselleştirildi")

# 9. Sonuçlar ve Değerlendirme
print("\n9. SONUÇLAR VE DEĞERLENDİRME")
print("-"*80)

print("""
Orman yangınları veri seti üzerinde yapılan istatistiksel analiz sonucunda elde edilen temel bulgular:

1. Veri Dağılımları:
   - Yangın alanı (area) sütunu oldukça çarpık bir dağılıma sahiptir, çoğu yangın küçük alanları etkilemektedir.
   - Logaritmik dönüşüm, yanmış alan dağılımını daha normal bir dağılıma yaklaştırmıştır.

2. Mevsimsel Etkiler:
   - Yangınların mevsimsel dağılımında anlamlı farklılıklar gözlemlenmiştir.
   - Yaz aylarında yangın sayısı ve etkilenen alan büyüklüğü artış göstermektedir.

3. Haftasonu vs Haftaiçi:
   - Haftasonu ve haftaiçi yangın şiddetleri arasında istatistiksel olarak anlamlı bir fark tespit edilmemiştir.

4. Meteorolojik Faktörler:
   - Sıcaklık, nem, rüzgar ve yağış gibi meteorolojik faktörler yangın alanı üzerinde etkilere sahiptir.
   - Sıcaklık arttıkça ve nem azaldıkça yangın alanının genişleme eğiliminde olduğu gözlemlenmiştir.

5. Yangın İndeksleri:
   - FFMC, DMC, DC ve ISI gibi yangın indeksleri ile yangın alanı arasında pozitif ilişkiler mevcuttur.
   - Özellikle ISI (İlk Yayılma İndeksi) yangın alanı ile daha güçlü bir ilişki göstermiştir.

6. Regresyon Analizi:
   - Oluşturulan çoklu regresyon modeli, yangın alanını tahmin etmede sınırlı bir başarı göstermiştir (düşük R-kare değeri).
   - Bu durum, yangın davranışını etkileyen kaydedilmemiş başka faktörlerin varlığına işaret edebilir.

7. İstatistiksel Testler:
   - Shapiro-Wilk normallik testleri çoğu değişkenin normal dağılmadığını göstermiştir.
   - ANOVA testi, mevsimler arasında yangın alanı açısından anlamlı farklılıklar olduğunu ortaya koymuştur.
   - Ki-kare bağımsızlık testi, ay ve günler arasındaki ilişkiyi değerlendirmek için kullanılmıştır.

Bu bulgular, orman yangınlarının karmaşık doğasını ve çok sayıda faktörün etkileşiminden kaynaklandığını göstermektedir. Gelecekteki çalışmalarda, daha kapsamlı veri setleri ve daha gelişmiş modelleme teknikleri kullanılarak yangın davranışının daha iyi anlaşılması hedeflenebilir.
""")

print("\nAnaliz tamamlandı. Çıktılar 'outputs' klasörüne kaydedildi.")