# =============================================================================
#  PREDIKSI HARGA RUMAH DI BANDUNG
#  Mini Project Data Science + Machine Learning
# =============================================================================
#  File ini berisi SELURUH pipeline dari awal hingga akhir:
#    Phase 1: Data Understanding & Cleaning
#    Phase 2: Feature Engineering & Preprocessing
#    Phase 3: Modeling & Evaluation
#    Phase 4: Save Model untuk Web App
#
#  Cara pakai:
#    1. Pastikan file Dataset/results_cleaned.csv ada
#    2. Jalankan: python main.py
#    3. Semua output (chart, model, data bersih) akan tersimpan otomatis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend agar bisa jalan tanpa GUI
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'webapp', 'model')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  PREDIKSI HARGA RUMAH DI BANDUNG")
print("  Mini Project Data Science")
print("=" * 60)

# =====================================================================
# PHASE 1: DATA UNDERSTANDING & CLEANING
# =====================================================================
print("\n[PHASE 1] DATA UNDERSTANDING & CLEANING")
print("-" * 60)

# --- 1.1 Load Data ---
df_raw = pd.read_csv(os.path.join(DATASET_DIR, 'results_cleaned.csv'))
print(f"  Data loaded: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
print(f"  Kolom: {list(df_raw.columns)}")
print(f"  Missing values: {df_raw.isnull().sum().sum()} (tidak ada)")

# --- 1.2 Cek Anomali ---
print(f"\n  Anomali ditemukan:")
print(f"    - Duplikat: {df_raw.duplicated(subset=['location','bedroom_count','bathroom_count','price','land_area','building_area (m2)']).sum()} baris")
print(f"    - Bedroom=0: {(df_raw['bedroom_count']==0).sum()} baris")
print(f"    - Bathroom=0: {(df_raw['bathroom_count']==0).sum()} baris")
print(f"    - Building area negatif: {(df_raw['building_area (m2)']<0).sum()} baris")
print(f"    - Lokasi unik: {df_raw['location'].nunique()} kecamatan")

# --- 1.3 Data Cleaning ---
df = df_raw.copy()
before_total = len(df)

# Step 1: Hapus duplikat
df = df.drop_duplicates(subset=['location', 'bedroom_count', 'bathroom_count',
                                 'price', 'land_area', 'building_area (m2)'])
print(f"\n  Cleaning:")
print(f"    1. Hapus duplikat: {before_total} -> {len(df)}")

# Step 2: Hapus bedroom/bathroom = 0 (bukan rumah)
before = len(df)
df = df[(df['bedroom_count'] > 0) & (df['bathroom_count'] > 0)]
print(f"    2. Hapus bedroom/bathroom=0: {before} -> {len(df)}")

# Step 3: Hapus building_area negatif/nol
before = len(df)
df = df[df['building_area (m2)'] > 0]
print(f"    3. Hapus building_area negatif: {before} -> {len(df)}")

# Step 4: Hapus data tidak masuk akal
before = len(df)
df = df[(df['building_area (m2)'] <= 10000) & (df['price'] >= 50_000_000)]
print(f"    4. Hapus data tidak masuk akal: {before} -> {len(df)}")

# --- 1.4 EDA: Visualisasi ---
print(f"\n  Membuat visualisasi EDA...")

# Chart 1: Distribusi Harga
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribusi Harga Rumah di Bandung', fontsize=16, fontweight='bold')
axes[0].hist(df['price'] / 1e9, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Harga (Miliar Rp)')
axes[0].set_ylabel('Jumlah Rumah')
axes[0].set_title('Histogram Harga')
axes[1].boxplot(df['price'] / 1e9, vert=True)
axes[1].set_ylabel('Harga (Miliar Rp)')
axes[1].set_title('Boxplot Harga')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_01_distribusi_harga.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Harga per Kecamatan
avg_price = df.groupby('location')['price'].mean().sort_values(ascending=True)
kecamatan_names = [loc.replace(', Bandung', '') for loc in avg_price.index]
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(kecamatan_names, avg_price.values / 1e9,
               color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(avg_price))))
ax.set_xlabel('Rata-rata Harga (Miliar Rp)')
ax.set_title('Rata-rata Harga Rumah per Kecamatan di Bandung', fontsize=14, fontweight='bold')
for bar, val in zip(bars, avg_price.values / 1e9):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}M', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_02_harga_per_kecamatan.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Correlation Heatmap
numeric_cols = ['bedroom_count', 'bathroom_count', 'carport_count',
                'price', 'land_area', 'building_area (m2)']
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax)
ax.set_title('Correlation Matrix - Fitur Numerik', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_03_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Hubungan Luas dengan Harga', fontsize=14, fontweight='bold')
axes[0].scatter(df['land_area'], df['price'] / 1e9, alpha=0.3, s=10, color='#2196F3')
axes[0].set_xlabel('Luas Tanah (m2)')
axes[0].set_ylabel('Harga (Miliar Rp)')
axes[0].set_title('Luas Tanah vs Harga')
axes[1].scatter(df['building_area (m2)'], df['price'] / 1e9, alpha=0.3, s=10, color='#FF5722')
axes[1].set_xlabel('Luas Bangunan (m2)')
axes[1].set_ylabel('Harga (Miliar Rp)')
axes[1].set_title('Luas Bangunan vs Harga')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_04_scatter_luas_vs_harga.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Distribusi Fitur
features_list = ['bedroom_count', 'bathroom_count', 'carport_count', 'land_area', 'building_area (m2)']
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Distribusi Fitur Numerik', fontsize=14, fontweight='bold')
for ax, col in zip(axes, features_list):
    ax.hist(df[col], bins=30, color='#4CAF50', edgecolor='white', alpha=0.8)
    ax.set_title(col.replace('_', '\n'), fontsize=10)
    ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_05_distribusi_fitur.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"    -> 5 chart EDA tersimpan di folder output/")

# Korelasi dengan harga
price_corr = corr['price'].drop('price').sort_values(ascending=False)
print(f"\n  Korelasi dengan harga:")
for feat, val in price_corr.items():
    print(f"    {feat}: {val:.3f}")
print(f"\n  Kecamatan termahal: {kecamatan_names[-1]}")
print(f"  Kecamatan termurah: {kecamatan_names[0]}")

# --- 1.5 Outlier Handling (IQR) ---
print(f"\n  Outlier Handling (IQR Method):")
print(f"    Data sebelum IQR: {len(df)} baris")

def remove_outliers_iqr(dataframe, columns, multiplier=1.5):
    df_clean = dataframe.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        print(f"      {col}: buang {before - len(df_clean)} baris (range: {lower:,.0f} - {upper:,.0f})")
    return df_clean

iqr_columns = ['price', 'land_area', 'building_area (m2)', 'bedroom_count', 'bathroom_count']
df_clean = remove_outliers_iqr(df, iqr_columns)
print(f"    Data setelah IQR: {len(df_clean)} baris")

# Chart 6: Before vs After IQR
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Boxplot: Sebelum vs Sesudah IQR', fontsize=14, fontweight='bold')
compare_cols = ['price', 'land_area', 'building_area (m2)']
for i, col in enumerate(compare_cols):
    axes[0, i].boxplot(df[col].values)
    axes[0, i].set_title(f'{col}\n(SEBELUM)')
    axes[1, i].boxplot(df_clean[col].values)
    axes[1, i].set_title(f'{col}\n(SESUDAH)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_06_before_after_iqr.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save data bersih
df_clean.to_csv(os.path.join(DATASET_DIR, 'data_final.csv'), index=False)
print(f"\n  Data bersih tersimpan: Dataset/data_final.csv ({len(df_clean)} baris)")

# =====================================================================
# PHASE 2: FEATURE ENGINEERING & PREPROCESSING
# =====================================================================
print(f"\n[PHASE 2] FEATURE ENGINEERING & PREPROCESSING")
print("-" * 60)

# --- 2.1 Feature Engineering ---
df_clean['building_ratio'] = df_clean['building_area (m2)'] / df_clean['land_area']
df_clean['total_rooms'] = df_clean['bedroom_count'] + df_clean['bathroom_count']
print(f"  Fitur baru: building_ratio, total_rooms")

# --- 2.2 Encoding Lokasi ---
le = LabelEncoder()
df_clean['location_encoded'] = le.fit_transform(df_clean['location'])
print(f"  Label Encoding: {len(le.classes_)} kecamatan")

# --- 2.3 Train-Test Split ---
feature_columns = ['location_encoded', 'bedroom_count', 'bathroom_count',
                   'carport_count', 'land_area', 'building_area (m2)',
                   'building_ratio', 'total_rooms']

X = df_clean[feature_columns]
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris")

# =====================================================================
# PHASE 3: MODELING & EVALUATION
# =====================================================================
print(f"\n[PHASE 3] MODELING & EVALUATION")
print("-" * 60)

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)
    return {
        'Model': name,
        'R2_Train': r2_score(y_tr, y_pred_train),
        'R2_Test': r2_score(y_te, y_pred_test),
        'MAE': mean_absolute_error(y_te, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred_test))
    }, model, y_pred_test

# Train 4 model
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42,
                                                    learning_rate=0.1, max_depth=5),
}

results = []
trained_models = {}

for name, model in models.items():
    metrics, trained_model, y_pred = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append(metrics)
    trained_models[name] = (trained_model, y_pred)
    print(f"\n  {name}:")
    print(f"    R2 Train={metrics['R2_Train']:.4f} | R2 Test={metrics['R2_Test']:.4f}")
    print(f"    MAE=Rp {metrics['MAE']:,.0f} | RMSE=Rp {metrics['RMSE']:,.0f}")

results_df = pd.DataFrame(results)
best_result = max(results, key=lambda x: x['R2_Test'])
best_name = best_result['Model']
best_model = trained_models[best_name][0]
best_pred = trained_models[best_name][1]

print(f"\n  >>> MODEL TERBAIK: {best_name} (R2 Test = {best_result['R2_Test']:.4f})")

# Chart 7: Perbandingan Model
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Perbandingan Performa Model', fontsize=14, fontweight='bold')
x_pos = np.arange(len(results_df))
axes[0].bar(x_pos - 0.15, results_df['R2_Train'], 0.3, label='Train', color='#2196F3', alpha=0.8)
axes[0].bar(x_pos + 0.15, results_df['R2_Test'], 0.3, label='Test', color='#FF5722', alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(results_df['Model'], rotation=15, ha='right')
axes[0].set_ylabel('R2 Score')
axes[0].set_title('R2 Score (Train vs Test)')
axes[0].legend()
axes[1].bar(results_df['Model'], results_df['MAE'] / 1e9, color='#4CAF50', alpha=0.8)
axes[1].set_ylabel('MAE (Miliar Rp)')
axes[1].set_title('Mean Absolute Error')
plt.setp(axes[1].get_xticklabels(), rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 8: Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test / 1e9, best_pred / 1e9, alpha=0.4, s=15, color='#2196F3')
lims = [0, max(y_test.max(), max(best_pred)) / 1e9]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Harga Aktual (Miliar Rp)')
ax.set_ylabel('Harga Prediksi (Miliar Rp)')
ax.set_title(f'Actual vs Predicted - {best_name}', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 9: Feature Importance
importances = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else trained_models['Random Forest'][0].feature_importances_
feat_imp = pd.DataFrame({'feature': feature_columns, 'importance': importances}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feat_imp['feature'], feat_imp['importance'], color='#2196F3', alpha=0.8)
ax.set_xlabel('Importance')
ax.set_title(f'Feature Importance - {best_name}', fontsize=14, fontweight='bold')
for bar, val in zip(bars, feat_imp['importance']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Feature Importance:")
for _, row in feat_imp.sort_values('importance', ascending=False).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# =====================================================================
# PHASE 4: SAVE MODEL & ARTIFACTS
# =====================================================================
print(f"\n[PHASE 4] SAVE MODEL & ARTIFACTS")
print("-" * 60)

joblib.dump(best_model, os.path.join(MODEL_DIR, 'house_price_model.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, 'feature_columns.pkl'))
joblib.dump(sorted(le.classes_.tolist()), os.path.join(MODEL_DIR, 'locations.pkl'))

stats = {
    'median_bedroom': int(df_clean['bedroom_count'].median()),
    'median_bathroom': int(df_clean['bathroom_count'].median()),
    'median_carport': int(df_clean['carport_count'].median()),
    'median_land_area': int(df_clean['land_area'].median()),
    'median_building_area': int(df_clean['building_area (m2)'].median()),
    'best_model_name': best_name,
    'r2_score': round(best_result['R2_Test'], 4),
    'mae': round(best_result['MAE']),
}
joblib.dump(stats, os.path.join(MODEL_DIR, 'stats.pkl'))
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'), index=False)

print(f"  Tersimpan di webapp/model/:")
print(f"    - house_price_model.pkl ({best_name})")
print(f"    - label_encoder.pkl")
print(f"    - feature_columns.pkl")
print(f"    - locations.pkl (27 kecamatan)")
print(f"    - stats.pkl")
print(f"\n  Tersimpan di output/:")
print(f"    - 9 chart visualisasi (.png)")
print(f"    - model_results.csv")

# =====================================================================
# RINGKASAN AKHIR
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  SELESAI! RINGKASAN PROJECT")
print(f"{'=' * 60}")
print(f"  Dataset awal:  {df_raw.shape[0]} baris")
print(f"  Dataset final: {len(df_clean)} baris")
print(f"  Model terbaik: {best_name}")
print(f"  R2 Score:      {best_result['R2_Test']:.4f}")
print(f"  MAE:           Rp {best_result['MAE']:,.0f}")
print(f"  RMSE:          Rp {best_result['RMSE']:,.0f}")
print(f"\n  Untuk menjalankan web app:")
print(f"    cd webapp")
print(f"    streamlit run app.py")
print(f"{'=' * 60}")
