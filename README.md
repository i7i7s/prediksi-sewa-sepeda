# 🚲 Simulator Prediksi Penyewaan Sepeda Harian

Aplikasi web interaktif (*dashboard*) untuk memprediksi jumlah penyewaan sepeda harian berdasarkan kondisi lingkungan, musim, dan parameter waktu. Project ini dibangun untuk memenuhi tugas Ujian Akhir Semester (UAS) mata kuliah Sains Data.

## 🌟 Fitur Utama
- **Hybrid Input System:** Kombinasi input otomatis berdasarkan bulan dan kontrol parameter cuaca manual menggunakan slider interaktif.
- **Konversi Skala Real (Human-Friendly):** Input di layar menggunakan satuan asli (°C, %, km/jam) yang secara otomatis dikonversi ke format normalisasi untuk kebutuhan model di balik layar.
- **Integrasi Model Machine Learning:** Didukung oleh algoritma *Gradient Boosting Regressor* (`scikit-learn`) yang telah dioptimalkan melalui tahapan *feature selection* untuk menghilangkan multikolinearitas.
- **Visualisasi Prediksi Dinamis:** Grafik Time Series interaktif menggunakan `Plotly` untuk membandingkan berbagai skenario prediksi buatan pengguna secara *real-time*.
- **Desain UI/UX Modern:** Mengusung tema *Light Mode Professional* khas dashboard SaaS modern.

## 🛠️ Teknologi yang Digunakan
- **Python 3**
- **Streamlit** (Web App Framework & Deployment)
- **Scikit-Learn** (Model Machine Learning)
- **Pandas & NumPy** (Manipulasi Data)
- **Plotly** (Visualisasi Data Interaktif)

## 🚀 Cara Menjalankan secara Lokal (Localhost)

1. Pastikan Anda sudah menginstall Python di komputer Anda.
2. *Clone* repository ini ke komputer Anda:
   ```bash
   git clone https://github.com/i7i7s/prediksi-sewa-sepeda.git
   cd prediksi-sewa-sepeda
   ```
3. Install seluruh *library* yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi menggunakan perintah Streamlit:
   ```bash
   streamlit run app.py
   ```

## 📂 Struktur Repositori
- `app.py` — File utama *(source code)* dari aplikasi Streamlit.
- `model_bike_sharing.pkl` — Model Machine Learning terlatih yang dibungkus dalam format *dictionary* (berisi *estimator* dan list *features*).
- `day.csv` — Dataset Capital Bikeshare harian sebagai data pendukung historis.
- `requirements.txt` — Daftar dependensi Python untuk proses *deployment* ke server.
- `.streamlit/config.toml` — Konfigurasi pengaturan tema paksa ke *Light Mode*.
