# Cancer Detection System 🏥

Sistem klasifikasi kanker menggunakan Machine Learning (SVM) dengan dataset histopatologi.

## 📋 Requirements

```bash
pip install streamlit scikit-learn scikit-image joblib pandas numpy matplotlib pillow plotly
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/zuzu101/Cancer_Detection.git
cd Cancer_Detection
```

### 2. Download Model Files

Karena model files terlalu besar untuk GitHub (767MB), download dari:

**Option A: Google Drive**
- [Download Model Files (ZIP)](LINK_GOOGLE_DRIVE_ANDA)

**Option B: Git LFS** (jika sudah di-setup)
```bash
git lfs pull
```

Letakkan files ini di root folder:
- `cancer_svm_model_20260110_235537.pkl`
- `cancer_scaler_20260110_235537.pkl`

### 3. Setup Dataset

Buat 3 folder dengan struktur:
```
Cancer_Detection/
├── Ganas/          # Gambar tumor ganas (100 images)
├── Jinak/          # Gambar tumor jinak (100 images)
├── Non Kanker/     # Gambar normal (84 images)
└── app.py
```

### 4. Run Application

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

## 📊 Features

- ✅ Upload dan prediksi gambar histopatologi
- ✅ Visualisasi confidence score
- ✅ Batch prediction dari folder
- ✅ Export hasil ke CSV
- ✅ Dashboard analytics
- ✅ Model evaluation metrics

## 🔧 Training Model (Optional)

Jika ingin training ulang model:

1. Siapkan dataset di folders Ganas/, Jinak/, Non Kanker/
2. Buka `biasalahPSD.ipynb` di Jupyter
3. Jalankan semua cells (5-10)
4. Model baru akan tersimpan dengan timestamp

## 📁 Project Structure

```
Cancer_Detection/
├── app.py                          # Streamlit application
├── biasalahPSD.ipynb              # Training notebook
├── cancer_svm_model_*.pkl         # Trained SVM model (767MB)
├── cancer_scaler_*.pkl            # Feature scaler
├── Ganas/                         # Cancer dataset (malignant)
├── Jinak/                         # Cancer dataset (benign)
└── Non Kanker/                    # Normal tissue dataset
```

## 🎯 Model Performance

- **Accuracy**: 73.68%
- **Model**: Support Vector Machine (SVM)
- **Features**: 150,528 (224x224x3 images)
- **Classes**: 3 (GANAS, JINAK, NON KANKER)
- **Training samples**: ~1,135 (with 4x augmentation)

## 📝 Notes

⚠️ **Important**: 
- Model files TIDAK di-push ke GitHub karena ukuran besar
- Download model files secara terpisah
- Dataset juga tidak included (privacy + ukuran besar)

## 🔗 Links

- GitHub: [zuzu101/Cancer_Detection](https://github.com/zuzu101/Cancer_Detection)
- Model Files: [Download Link]
