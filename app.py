import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tslearn


# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Bone Age Assessment",
    page_icon="🦴",
    layout="wide"
)

# ==========================================
# 2. FUNGSI LOAD MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_model_and_encoder():
    # Pastikan file .pkl ini ada di satu folder dengan app.py
    # Ganti nama file sesuai dengan yang Anda simpan di notebook
    try:
        model = joblib.load('models/model1/best_model_bone_age.pkl') 
        le = joblib.load('models/model1/label_encoder.pkl')
        return model, le
    except FileNotFoundError:
        st.error("File Model tidak ditemukan! Pastikan 'best_model_bone_age.pkl' dan 'label_encoder.pkl' sudah ada.")
        return None, None

model, le = load_model_and_encoder()

# ==========================================
# 3. JUDUL DAN SIDEBAR
# ==========================================
st.title("🦴 Sistem Deteksi Usia Tulang (Distal Phalanx)")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning** untuk memprediksi kelompok usia berdasarkan 
bentuk outline tulang jari (Distal Phalanx).
""")

# ==========================================
# 4. AREA UPLOAD FILE
# ==========================================
uploaded_file = st.file_uploader("Upload File Outline (.txt)", type=["txt", "csv"])

if uploaded_file is not None and model is not None:
    try:
        # --- MEMBACA DATA ---
        # sep='\s+' menangani spasi ganda/tunggal pada file .txt
        # engine='python' agar lebih stabil membaca separator regex
        df = pd.read_csv(uploaded_file, sep='\s+', header=None, engine='python')
        
        st.subheader("1. Preview Data")
        st.write(f"Ukuran Data: {df.shape[0]} baris, {df.shape[1]} kolom")
        st.dataframe(df.head())

        # --- PREPROCESSING OTOMATIS ---
        # Cek apakah file memiliki kolom label (biasanya kolom ke-1 jadi 81 kolom total)
        # Jika kolom > 80, kita asumsikan kolom 0 adalah label lama, jadi kita buang.
        if df.shape[1] > 80:
            st.warning("Mendeteksi kolom label di kolom pertama. Kolom tersebut akan diabaikan untuk prediksi.")
            X_input = df.iloc[:, 1:].values # Ambil kolom 1 sampai akhir
        else:
            X_input = df.values # Ambil semua

        # --- TOMBOL PREDIKSI ---
        if st.button("Jalankan Prediksi"):
            st.subheader("2. Hasil Analisis")
            
            # Persiapan input model
            # Jika menggunakan KNN-DTW (tslearn), bentuk harus 3D (n_samples, 80, 1)
            # Kita cek tipe modelnya atau kita coba reshape default aman
            if "tslearn" in str(type(model)):
                X_final = X_input.reshape((X_input.shape[0], X_input.shape[1], 1))
            else:
                X_final = X_input # Untuk Random Forest / KNN biasa

            # Prediksi
            y_pred_code = model.predict(X_final)
            
            # Kembalikan ke Label Asli (inverse transform)
            y_pred_label = le.inverse_transform(y_pred_code)

            # Tampilkan Hasil per Baris
            # Kita buat DataFrame hasil agar rapi
            results_df = pd.DataFrame({
                "Data Ke-": range(1, len(y_pred_label) + 1),
                "Prediksi Kode": y_pred_code,
                "Prediksi Kelompok Usia (Label)": y_pred_label
            })
            
            st.table(results_df)

# --- VISUALISASI ---
            st.subheader("3. Visualisasi Bentuk Tulang")
            
            jumlah_data = len(X_input)
            
            if jumlah_data > 1:
                # Jika data lebih dari 1, tampilkan slider
                row_to_show = st.slider("Pilih indeks data untuk dilihat grafiknya:", 0, jumlah_data - 1, 0)
            else:
                # Jika data cuma 1, langsung set ke 0 tanpa slider
                st.info("Menampilkan visualisasi untuk satu-satunya data yang tersedia.")
                row_to_show = 0
            
            # Plotting Grafik
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Pastikan mengambil data yang benar (X_input bisa berupa array 2D atau 3D tergantung preprocessing sebelumnya)
            # Kita ratakan (flatten) agar aman saat di-plot
            data_garis = X_input[row_to_show].flatten()
            
            label_prediksi = y_pred_label[row_to_show]
            
            ax.plot(data_garis, label=f'Data Input (Prediksi: {label_prediksi})', color='blue', linewidth=2)
            ax.set_title(f"Visualisasi Outline Tulang - Data ke-{row_to_show + 1}")
            ax.set_xlabel("Titik Urutan Outline")
            ax.set_ylabel("Nilai Posisi")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        st.write("Pastikan format isi file .txt sesuai standar dataset (dipisahkan spasi).")