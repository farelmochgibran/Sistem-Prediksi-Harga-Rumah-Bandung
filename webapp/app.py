# =============================================================================
# STREAMLIT WEB APP - Prediksi Harga Rumah di Bandung
# Jalankan: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Rumah Bandung",
    page_icon="🏠",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1a1a2e;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-price {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Artifacts ---
@st.cache_resource
def load_model():
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    model = joblib.load(os.path.join(model_dir, 'house_price_model.pkl'))
    le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
    locations = joblib.load(os.path.join(model_dir, 'locations.pkl'))
    stats = joblib.load(os.path.join(model_dir, 'stats.pkl'))
    return model, le, feature_cols, locations, stats

model, le, feature_cols, locations, stats = load_model()

# --- Header ---
st.markdown('<p class="main-title">🏠 Prediksi Harga Rumah di Bandung</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Masukkan spesifikasi rumah untuk mendapatkan estimasi harga</p>', unsafe_allow_html=True)

st.divider()

# --- Sidebar: Info Model ---
with st.sidebar:
    st.header("📊 Informasi Model")
    st.metric("Model", stats['best_model_name'])
    st.metric("R² Score", f"{stats['r2_score']:.4f}")
    st.metric("MAE", f"Rp {stats['mae']:,.0f}")

    st.divider()

    st.header("📍 Daftar Kecamatan")
    st.caption("27 kecamatan di Bandung tersedia:")
    for loc in locations:
        kec = loc.replace(", Bandung", "")
        st.text(f"• {kec}")

# --- Form Input ---
st.subheader("📝 Input Spesifikasi Rumah")

col1, col2 = st.columns(2)

with col1:
    # Lokasi (WAJIB)
    location_display = [loc.replace(", Bandung", "") for loc in locations]
    selected_display = st.selectbox(
        "📍 Lokasi (Kecamatan) *",
        options=location_display,
        index=0,
        help="Pilih kecamatan di Bandung"
    )
    selected_location = f"{selected_display}, Bandung"

    # Luas Tanah (WAJIB)
    land_area = st.number_input(
        "📐 Luas Tanah (m²) *",
        min_value=10,
        max_value=1000,
        value=stats['median_land_area'],
        step=10,
        help="Luas tanah dalam meter persegi"
    )

    # Luas Bangunan (OPSIONAL)
    building_area = st.number_input(
        "🏗️ Luas Bangunan (m²)",
        min_value=10,
        max_value=1000,
        value=stats['median_building_area'],
        step=10,
        help="Luas bangunan dalam meter persegi"
    )

with col2:
    # Kamar Tidur (OPSIONAL)
    bedroom = st.number_input(
        "🛏️ Jumlah Kamar Tidur",
        min_value=1,
        max_value=10,
        value=stats['median_bedroom'],
        step=1
    )

    # Kamar Mandi (OPSIONAL)
    bathroom = st.number_input(
        "🚿 Jumlah Kamar Mandi",
        min_value=1,
        max_value=10,
        value=stats['median_bathroom'],
        step=1
    )

    # Carport (OPSIONAL)
    carport = st.number_input(
        "🚗 Jumlah Carport",
        min_value=0,
        max_value=10,
        value=stats['median_carport'],
        step=1
    )

st.divider()

# --- Prediksi ---
if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
    # Encode lokasi
    location_encoded = le.transform([selected_location])[0]

    # Hitung fitur tambahan
    building_ratio = building_area / land_area if land_area > 0 else 1.0
    total_rooms = bedroom + bathroom

    # Susun fitur sesuai urutan training
    features = pd.DataFrame({
        'location_encoded': [location_encoded],
        'bedroom_count': [bedroom],
        'bathroom_count': [bathroom],
        'carport_count': [carport],
        'land_area': [land_area],
        'building_area (m2)': [building_area],
        'building_ratio': [building_ratio],
        'total_rooms': [total_rooms],
    })

    # Prediksi
    prediction = model.predict(features)[0]

    # Pastikan harga tidak negatif
    prediction = max(prediction, 0)

    # Format Rupiah
    def format_rupiah(value):
        if value >= 1_000_000_000:
            return f"Rp {value/1_000_000_000:,.2f} Miliar"
        elif value >= 1_000_000:
            return f"Rp {value/1_000_000:,.0f} Juta"
        else:
            return f"Rp {value:,.0f}"

    # Tampilkan hasil
    st.markdown(f"""
    <div class="result-box">
        <p style="font-size: 1rem; margin-bottom: 0.5rem;">Estimasi Harga Rumah</p>
        <p class="result-price">{format_rupiah(prediction)}</p>
        <p style="font-size: 0.85rem; opacity: 0.8;">
            {selected_display} • {land_area} m² tanah • {building_area} m² bangunan
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Detail input
    st.subheader("📋 Detail Input")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("Lokasi", selected_display)
    detail_col1.metric("Luas Tanah", f"{land_area} m²")
    detail_col2.metric("Luas Bangunan", f"{building_area} m²")
    detail_col2.metric("Kamar Tidur", bedroom)
    detail_col3.metric("Kamar Mandi", bathroom)
    detail_col3.metric("Carport", carport)

    # Disclaimer
    st.info(
        "⚠️ **Disclaimer:** Prediksi ini adalah estimasi berdasarkan data historis "
        "dan model machine learning. Harga aktual dapat berbeda tergantung kondisi pasar, "
        "kondisi bangunan, akses jalan, dan faktor lainnya."
    )

# --- Footer ---
st.divider()
st.caption("🎓 Mini Project Data Science — Prediksi Harga Rumah di Bandung")
st.caption(f"Model: {stats['best_model_name']} | R² = {stats['r2_score']:.4f} | Data: 3,905 rumah")
