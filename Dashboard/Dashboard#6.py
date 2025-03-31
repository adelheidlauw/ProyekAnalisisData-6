import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ------------------------------
# 📌 Judul Aplikasi
st.title("🌤️ Analisis Polusi Udara - Kota Wanliu 🌧️")

# ------------------------------
# Membaca File
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "PRSA_Data_Wanliu_20130301-20170228.csv")
    Wanliu_df = pd.read_csv(file_path)
    Wanliu_df['date'] = pd.to_datetime(Wanliu_df[['year', 'month', 'day']])
    Wanliu_df['year_only'] = Wanliu_df['date'].dt.year
    return Wanliu_df

Wanliu_df = load_data()

# ------------------------------
# Sidebar Filter Tahun
st.sidebar.header("🔍 Pilih Tahun")
tahun = Wanliu_df['year_only'].unique()
tahun_dipilih = st.sidebar.selectbox("Pilih Tahun", tahun)

# Filter data berdasarkan tahun yang dipilih
Wanliu_df['year_only'] = Wanliu_df['date'].dt.year
avg_pm10_per_year = Wanliu_df.groupby(['year_only', 'date'])['PM10'].mean().reset_index()
filtered_Wanliu = Wanliu_df[Wanliu_df['year_only'] == tahun_dipilih].copy()

# ------------------------------
# Menampilkan DataFrame
st.subheader("📌 DataFrame yang Dipilih")

# Membuat expander untuk menampilkan data
with st.expander(" **DataFrame**", expanded=False):
    st.dataframe(filtered_Wanliu)

# Statistik Deskriptif
st.subheader("📌 Statistik Deskriptif")

# Membuat expander
with st.expander("**Rangkuman Parameter Statistik**", expanded=False):
    st.write(filtered_Wanliu.describe())

# ------------------------------
# Exploratory Data Analysis (EDA)
st.subheader("📌 Exploratory Data Analysis (EDA)")

kolom_numerik_eda = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
# Deteksi Outlier dengan IQR
Q1_eda = filtered_Wanliu[kolom_numerik_eda].quantile(0.25)
Q3_eda = filtered_Wanliu[kolom_numerik_eda].quantile(0.75)
IQR_eda = Q3_eda - Q1_eda

batas_bawah = Q1_eda - 1.5 * IQR_eda
batas_atas = Q3_eda + 1.5 * IQR_eda

clean_df = filtered_Wanliu[~((filtered_Wanliu[kolom_numerik_eda] < batas_bawah) | (filtered_Wanliu[kolom_numerik_eda] > batas_atas)).any(axis=1)]

# Cek jumlah data sebelum dan sesudah
st.write(f"Jumlah data sebelum pembersihan: {filtered_Wanliu.shape[0]}")
st.write(f"Jumlah data setelah pembersihan: {clean_df.shape[0]}")

with st.expander("**DataFrame yang Sudah Dibersihkan**", expanded=False):
    st.write(clean_df.describe())

# Visualisasi Distribusi Data Kotor
st.write("### Distribusi Data")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cols = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'WSPM']
for ax, col in zip(axes.flatten(), cols):
    sns.histplot(filtered_Wanliu[col], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribusi {col}')
plt.tight_layout()

# Membuat expander data kotor
with st.expander("### Distribusi Data Sebelum Pembersihan", expanded=False):
    st.pyplot(fig)

# Visualisasi Distribusi Data Bersih
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cols = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'WSPM']
for ax, col in zip(axes.flatten(), cols):
    sns.histplot(clean_df[col], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribusi {col}')
plt.tight_layout()

# Membuat expander data bersih
with st.expander("### Distribusi Data Setelah Pembersihan", expanded=False):
    st.pyplot(fig)

# Boxplot untuk Outlier
st.write("### Deteksi Outlier dengan Boxplot")
# Boxplot Data Kotor
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered_Wanliu[cols])
plt.xticks(rotation=45)
plt.title('Boxplot untuk Mendeteksi Outlier pada Data Sebelum Pembersihan')
#Membuat expander  data kotor
with st.expander("### Deteksi Outlier dengan Boxplot pada Data Sebelum Pembersihan",expanded=False):
    st.pyplot(fig)

# Boxplot Data Bersih
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=clean_df[cols])
plt.xticks(rotation=45)
plt.title('Boxplot untuk Mendeteksi Outlier pada Data Sesudah Pembersihan')
#Membuat expander  data bersih
with st.expander("### Deteksi Outlier dengan Boxplot pada Data Sesudah Pembersihan",expanded=False):
    st.pyplot(fig)

# Filter data berdasarkan tahun yang dipilih
filtered_data = avg_pm10_per_year[avg_pm10_per_year['year_only'] == tahun_dipilih]

# ------------------------------
# Visualization dan Explanatory Analysis
st.subheader("📌 Visualization dan Explanatory Analysis")
# ------------------------------
st.subheader(f"📈 Rata-Rata PM10 pada Tahun {tahun_dipilih}")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=filtered_data, x='date', y='PM10', marker='o', ax=ax)
plt.title(f"Tren Rata-Rata Harian PM10 pada Tahun {tahun_dipilih}")
plt.xlabel("Tanggal")
plt.ylabel("Rata-Rata PM10")
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig)


# ------------------------------
# Heatmap Korelasi
st.subheader("🔥 Heatmap Korelasi")
st.write("Visualisasi hubungan antar variabel dengan heatmap korelasi")
fig, ax = plt.subplots(figsize=(10, 6))
corr = clean_df[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)
