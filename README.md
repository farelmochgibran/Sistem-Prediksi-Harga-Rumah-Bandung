# 🏠 Prediksi Harga Rumah di Bandung

Mini project Data Science + Web App untuk memprediksi harga rumah di Kota Bandung menggunakan Machine Learning.

## 📋 Deskripsi

Project ini membangun model prediksi harga rumah berdasarkan fitur-fitur seperti lokasi (kecamatan), luas tanah, luas bangunan, jumlah kamar, dan carport. Model terbaik kemudian di-deploy sebagai web app interaktif menggunakan Streamlit.

## 📊 Dataset

| Info | Detail |
|------|--------|
| Sumber | Data listing rumah di Bandung |
| Data mentah | 7,609 baris |
| Data setelah cleaning | 3,905 baris |
| Fitur | 8 kolom (6 numerik + 1 kategorikal + 1 teks) |
| Cakupan | 27 kecamatan di Bandung |

### Fitur Dataset

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `house_name` | Text | Nama/judul listing rumah |
| `location` | Kategorikal | Kecamatan, Bandung |
| `bedroom_count` | Numerik | Jumlah kamar tidur |
| `bathroom_count` | Numerik | Jumlah kamar mandi |
| `carport_count` | Numerik | Jumlah carport |
| `land_area` | Numerik | Luas tanah (m²) |
| `building_area` | Numerik | Luas bangunan (m²) |
| `price` | Numerik | Harga rumah (Rp) — **TARGET** |

## 🔬 Metodologi

### 1. Data Cleaning
- Hapus 1,751 baris duplikat
- Buang listing non-rumah (bedroom/bathroom = 0)
- Hapus data error (building_area negatif)
- Outlier handling dengan metode **IQR (Interquartile Range)**

### 2. Feature Engineering
- `building_ratio` = building_area / land_area
- `total_rooms` = bedroom_count + bathroom_count
- Label Encoding untuk kolom lokasi

### 3. Modeling

| Model | R² Train | R² Test | MAE (Rp) | RMSE (Rp) |
|-------|----------|---------|----------|-----------|
| Linear Regression | 0.6636 | 0.7106 | 635 juta | 972 juta |
| Decision Tree | 0.9992 | 0.6378 | 646 juta | 1.08 miliar |
| **Random Forest** | **0.9631** | **0.7978** | **513 juta** | **813 juta** |
| Gradient Boosting | 0.9419 | 0.7901 | 516 juta | 828 juta |

**Model terbaik: Random Forest** (R² = 0.7978)

### 4. Feature Importance

| Ranking | Fitur | Importance |
|---------|-------|------------|
| 1 | land_area | 0.5396 |
| 2 | building_area | 0.2312 |
| 3 | location | 0.0830 |
| 4 | building_ratio | 0.0669 |
| 5 | carport_count | 0.0248 |

## 🌐 Web App

Web app interaktif menggunakan **Streamlit** yang memungkinkan user:
- Memilih lokasi (27 kecamatan)
- Input luas tanah dan bangunan
- Input jumlah kamar (opsional)
- Mendapatkan estimasi harga

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Jalankan Script ML (opsional)
```bash
python main.py    # Menjalankan seluruh pipeline: Data Cleaning, EDA, Feature Engineering & Modeling
```

### 3. Jalankan Web App
```bash
cd webapp
streamlit run app.py
```
Buka browser di `http://localhost:8501`

## 📁 Struktur Project

```
Metopen/
├── Dataset/
│   ├── results_cleaned.csv     # Dataset asli
│   └── data_final.csv          # Dataset setelah cleaning
├── output/                     # Hasil visualisasi EDA & evaluasi model
├── webapp/
│   ├── app.py                  # Streamlit web app
│   └── model/
│       ├── house_price_model.pkl
│       ├── label_encoder.pkl
│       ├── feature_columns.pkl
│       ├── locations.pkl
│       └── stats.pkl
├── docs/
│   └── PLAN-house-price-prediction.md
├── main.py                     # Script utama pipeline Machine Learning
├── requirements.txt            # Project dependencies
└── README.md
```

## 💡 Insight

- **Kecamatan termahal:** Bandung Wetan
- **Kecamatan termurah:** Panyileukan
- **Fitur paling berpengaruh:** Luas tanah (54%) dan luas bangunan (23%)
- Lokasi menyumbang ~8% terhadap prediksi harga

## 🔮 Potensi Pengembangan

- Filter berdasarkan budget (rekomendasikan lokasi sesuai budget)
- Peta interaktif (Folium) menampilkan harga per kecamatan
- Tambah data jarak ke pusat kota / fasilitas umum
- Deploy ke Streamlit Cloud untuk akses publik
- Eksperimen model Deep Learning (Neural Network)

## ⚠️ Limitations

- Model IQR dihitung dari seluruh data (idealnya hanya dari train set)
- Label Encoding mengasumsikan ordinal pada lokasi (OK untuk tree-based model)
- Data hanya dari satu sumber listing, mungkin bias terhadap harga listing vs harga jual
- R² = 0.80 berarti masih ada ~20% variasi harga yang tidak bisa dijelaskan model

## 🎓 Tech Stack

Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Streamlit • Joblib
"# Sistem-Prediksi-Harga-Rumah-Bandung" 
