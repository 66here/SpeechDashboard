# Suara & Emosi — Dashboard SER IndoWaveSentiment

    pip install -r requirements.txt
    streamlit run app.py        # jalankan dari folder ini

Halaman: Overview · Emotion Prediction · Model Performance · Error Analysis · Dataset & Methodology.

- Model: `model/model_final.joblib` (MFCC + SVM, base), dimuat lewat `src/infer.py`, fitur dari `src/preprocess_mfcc.py` (tidak diubah).
- Semua angka dibaca dari `assets/*.csv`; tidak ada data atau hasil buatan.
- `app_utils.py` berisi fungsi data/metrik tanpa Streamlit; `app.py` berisi tema, grafik, dan halaman.
- Aset opsional yang hilang tidak menghentikan aplikasi; bagian terkait menampilkan pesan "tidak tersedia".
