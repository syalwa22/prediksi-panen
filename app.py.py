import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("rf_model.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

le_pupuk = joblib.load("label_encoder_pupuk.pkl")
le_padi = joblib.load("label_encoder_padi.pkl")

# ==========================
# NILAI EFEKTIVITAS PUPUK
# ==========================

nilai_pupuk_map = {
    "CMP 36, NPK Ponska": 1.0,
    "KCL, NPK Ponska": 1.1,
    "KSP, NPK Ponska": 0.9,
    "NPK12-12": 1.3,
    "SCS 36, NPK Ponska": 1.0,
    "Urea N46, NPK Ponska": 1.2,
    "Urea Nitrea, NPK Ponska": 1.1
}

# ==========================
# JUDUL APLIKASI
# ==========================

st.title("🌾 Prediksi Hasil Panen Padi")
st.write("Kelompok Tani Panca Karya")

# ==========================
# INPUT USER
# ==========================

luas_lahan = st.number_input(
    "Luas Lahan (Ha)",
    min_value=0.01,
    value=0.50
)

jumlah_pupuk = st.number_input(
    "Jumlah Pupuk (Kg)",
    min_value=1.0,
    value=100.0
)

jenis_pupuk = st.selectbox(
    "Jenis Pupuk",
    le_pupuk.classes_
)

curah_hujan = st.number_input(
    "Curah Hujan (mm)",
    min_value=0.0,
    value=400.0
)

suhu = st.number_input(
    "Suhu (°C)",
    min_value=0.0,
    value=27.0
)

jenis_padi = st.selectbox(
    "Jenis Padi",
    le_padi.classes_
)

# ==========================
# TOMBOL PREDIKSI
# ==========================

if st.button("Prediksi"):

    # ----------------------
    # Encoding
    # ----------------------

    jenis_pupuk_enc = le_pupuk.transform([jenis_pupuk])[0]
    jenis_padi_enc = le_padi.transform([jenis_padi])[0]

    # ----------------------
    # Nilai Pupuk
    # ----------------------

    nilai_pupuk = nilai_pupuk_map[jenis_pupuk]

    # ----------------------
    # Feature Engineering
    # ----------------------

    pupuk_per_lahan = jumlah_pupuk / luas_lahan

    pupuk_effect = jumlah_pupuk * nilai_pupuk

    luas_lahan_sq = luas_lahan ** 2

    jumlah_pupuk_log = np.log1p(jumlah_pupuk)

    pupuk_per_lahan_log = np.log1p(pupuk_per_lahan)

    interaksi_lahan_pupuk = luas_lahan * jumlah_pupuk

    # ----------------------
    # Dataset Prediksi
    # ----------------------

    fitur = pd.DataFrame([[
        luas_lahan,
        jumlah_pupuk,
        jenis_pupuk_enc,
        curah_hujan,
        suhu,
        jenis_padi_enc,
        nilai_pupuk,
        pupuk_per_lahan,
        pupuk_effect,
        luas_lahan_sq,
        jumlah_pupuk_log,
        pupuk_per_lahan_log,
        interaksi_lahan_pupuk
    ]],
    columns=[
        'luas_lahan',
        'jumlah_pupuk',
        'jenis_pupuk',
        'curah_hujan',
        'suhu',
        'jenis_padi',
        'nilai_pupuk',
        'pupuk_per_lahan',
        'pupuk_effect',
        'luas_lahan_sq',
        'jumlah_pupuk_log',
        'pupuk_per_lahan_log',
        'interaksi_lahan_pupuk'
    ])

    # ----------------------
    # Normalisasi
    # ----------------------

    fitur_scaled = scaler_X.transform(fitur)

    # ----------------------
    # Prediksi
    # ----------------------

    pred_scaled = model.predict(fitur_scaled)

    hasil_panen = scaler_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    )[0][0]

    # ==========================
        # PRODUKTIVITAS
    # ==========================

    produktivitas = hasil_panen / luas_lahan


    # ==========================
    # TAMPILKAN HASIL
    # ==========================

    st.success(
        f"🌾 Prediksi Hasil Panen: {hasil_panen:.2f} Kg"
    )

   