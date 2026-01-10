import streamlit as st
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Cancer Classification System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: #e0e7ff;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-label {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Stats box */
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .stat-value {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    /* Alert boxes */
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# LOAD MODEL FUNCTION
# ===========================
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        # Cari file model terbaru
        model_files = [f for f in os.listdir('.') if f.startswith('cancer_svm_model_') and f.endswith('.pkl')]
        scaler_files = [f for f in os.listdir('.') if f.startswith('cancer_scaler_') and f.endswith('.pkl')]
        
        if not model_files or not scaler_files:
            return None, None, "Model files not found! Please train the model first."
        
        # Ambil file terbaru
        model_file = sorted(model_files)[-1]
        scaler_file = sorted(scaler_files)[-1]
        
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        return model, scaler, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

# ===========================
# PREDICTION FUNCTION
# ===========================
def predict_image(image, model, scaler):
    """Predict cancer type from image"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Handle grayscale
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Resize to 224x224x3
        img_resized = resize(img_array, (224, 224, 3))
        
        # Normalize [0, 1]
        img_normalized = img_resized.astype(np.float32)
        if img_normalized.max() > 1:
            img_normalized = img_normalized / 255.0
        
        # Flatten and reshape
        img_flat = img_normalized.flatten().reshape(1, -1)
        
        # Scale
        img_scaled = scaler.transform(img_flat)
        
        # Predict
        prediction = model.predict(img_scaled)[0]
        probabilities = model.predict_proba(img_scaled)[0]
        
        categories = ['GANAS', 'JINAK', 'NON KANKER']
        
        return {
            'success': True,
            'class': categories[prediction],
            'class_index': int(prediction),
            'confidence': float(probabilities[prediction] * 100),
            'probabilities': {cat: float(prob*100) for cat, prob in zip(categories, probabilities)},
            'processed_image': img_resized
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ===========================
# MAIN APP
# ===========================
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏥 Cancer Classification System</h1>
        <p class="header-subtitle">Advanced AI-Powered Medical Image Analysis using Support Vector Machine</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, error = load_model_and_scaler()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("💡 Please run the training notebook first to generate the model files.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital-3.png", width=80)
        st.markdown("### 📊 System Information")
        
        st.markdown("""
        <div class="stat-box">
            <p class="stat-label">Model Type</p>
            <p class="stat-value">SVM Linear</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-box">
            <p class="stat-label">Image Resolution</p>
            <p class="stat-value">224×224×3</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-box">
            <p class="stat-label">Classes</p>
            <p class="stat-value">3 Types</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Classification Types")
        st.markdown("""
        - 🔴 **GANAS** - Malignant (Cancerous)
        - 🟡 **JINAK** - Benign (Non-cancerous tumor)
        - 🟢 **NON KANKER** - Non-cancer (Healthy)
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses **Support Vector Machine (SVM)** 
        with linear kernel trained on medical imaging dataset.
        
        **Features:**
        - Real-time prediction
        - Confidence scoring
        - Professional visualization
        - High accuracy classification
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Image Classification", "📈 Batch Processing", "📚 Information"])
    
    # TAB 1: Single Image Classification
    with tab1:
        st.markdown("### Upload Medical Image for Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file (JPG, PNG, JPEG)",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a medical image for cancer classification"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Predict button
                if st.button("🔬 Analyze Image", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        result = predict_image(image, model, scaler)
                    
                    if result['success']:
                        # Store result in session state
                        st.session_state['last_result'] = result
                    else:
                        st.error(f"❌ Error: {result['error']}")
        
        with col2:
            if 'last_result' in st.session_state:
                result = st.session_state['last_result']
                
                # Prediction result
                class_colors = {
                    'GANAS': '#dc3545',
                    'JINAK': '#ffc107', 
                    'NON KANKER': '#28a745'
                }
                
                class_icons = {
                    'GANAS': '🔴',
                    'JINAK': '🟡',
                    'NON KANKER': '🟢'
                }
                
                predicted_class = result['class']
                
                st.markdown(f"""
                <div style="background: {class_colors[predicted_class]}; color: white; 
                            padding: 2rem; border-radius: 10px; text-align: center;">
                    <p style="font-size: 1.2rem; margin: 0;">Prediction Result</p>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">
                        {class_icons[predicted_class]} {predicted_class}
                    </h1>
                    <p style="font-size: 1.5rem; margin: 0;">
                        Confidence: {result['confidence']:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("### 📊 Probability Distribution")
                
                probs = result['probabilities']
                df_probs = pd.DataFrame({
                    'Class': list(probs.keys()),
                    'Probability (%)': list(probs.values())
                })
                
                # Plotly bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_probs['Class'],
                        y=df_probs['Probability (%)'],
                        marker_color=['#dc3545', '#ffc107', '#28a745'],
                        text=[f"{v:.1f}%" for v in df_probs['Probability (%)']],
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Class Probabilities",
                    xaxis_title="Cancer Type",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 105],
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed probabilities
                st.markdown("### 📋 Detailed Analysis")
                for class_name, prob in probs.items():
                    icon = class_icons[class_name]
                    st.markdown(f"**{icon} {class_name}**: {prob:.2f}%")
                    st.progress(prob / 100)
    
    # TAB 2: Batch Processing
    with tab2:
        st.markdown("### Batch Image Processing")
        st.info("📁 Upload multiple images for batch classification")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple medical images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} images uploaded**")
            
            if st.button("🚀 Process All Images", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    
                    image = Image.open(file)
                    result = predict_image(image, model, scaler)
                    
                    if result['success']:
                        results.append({
                            'Filename': file.name,
                            'Prediction': result['class'],
                            'Confidence (%)': f"{result['confidence']:.2f}",
                            'GANAS (%)': f"{result['probabilities']['GANAS']:.2f}",
                            'JINAK (%)': f"{result['probabilities']['JINAK']:.2f}",
                            'NON KANKER (%)': f"{result['probabilities']['NON KANKER']:.2f}"
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                
                # Display results
                st.markdown("### 📊 Batch Results")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Download results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"cancer_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                st.markdown("### 📈 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ganas_count = len(df_results[df_results['Prediction'] == 'GANAS'])
                    st.metric("🔴 GANAS", ganas_count)
                
                with col2:
                    jinak_count = len(df_results[df_results['Prediction'] == 'JINAK'])
                    st.metric("🟡 JINAK", jinak_count)
                
                with col3:
                    non_kanker_count = len(df_results[df_results['Prediction'] == 'NON KANKER'])
                    st.metric("🟢 NON KANKER", non_kanker_count)
    
    # TAB 3: Information
    with tab3:
        st.markdown("### 📚 System Documentation")
        
        st.markdown("""
        ## 🎯 About This System
        
        This **Cancer Classification System** uses advanced Machine Learning techniques 
        to classify medical images into three categories:
        
        - **GANAS (Malignant)**: Cancerous tumors that can spread
        - **JINAK (Benign)**: Non-cancerous tumors
        - **NON KANKER (Non-cancer)**: Healthy tissue
        
        ## 🔬 Technology Stack
        
        - **Algorithm**: Support Vector Machine (SVM) with Linear Kernel
        - **Image Processing**: scikit-image
        - **Framework**: Streamlit
        - **Visualization**: Plotly, Matplotlib
        - **Model Training**: scikit-learn
        
        ## 📊 Model Specifications
        
        - **Input Size**: 224×224×3 RGB images
        - **Features**: 150,528 features per image
        - **Preprocessing**: 
          - Automatic grayscale to RGB conversion
          - Image normalization [0, 1]
          - Standard scaling with StandardScaler
        - **Training**: 
          - Data augmentation (4x factor)
          - Rotation, flip, brightness adjustments
          - Class weight balancing
        
        ## 🚀 How to Use
        
        ### Single Image Classification:
        1. Go to the "Image Classification" tab
        2. Upload a medical image (JPG, PNG)
        3. Click "Analyze Image"
        4. View prediction results and confidence scores
        
        ### Batch Processing:
        1. Go to the "Batch Processing" tab
        2. Upload multiple images
        3. Click "Process All Images"
        4. Download results as CSV
        
        ## ⚠️ Important Notes
        
        - This system is for **research and educational purposes**
        - Always consult with medical professionals for diagnosis
        - Results should be verified by qualified healthcare providers
        - Not intended as a replacement for professional medical advice
        
        ## 📞 Support
        
        For questions or issues, please contact the development team.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b;">
            <p>Developed with ❤️ using Python & Streamlit</p>
            <p>© 2026 Cancer Classification System</p>
        </div>
        """, unsafe_allow_html=True)

# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pickle
from datetime import datetime

# Konfigurasi Streamlit
st.set_page_config(
    page_title="SVM Cancer Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏥 Klasifikasi Kanker dengan SVM")
st.markdown("Aplikasi untuk mengklasifikasi kanker menggunakan Support Vector Machine (SVM)")

# Sidebar
with st.sidebar:
    st.header("📋 Menu")
    page = st.radio("Pilih Halaman:", 
                    ["🏠 Dashboard", 
                     "📊 Data Loading", 
                     "🤖 Training Model", 
                     "📈 Hasil & Evaluasi",
                     "🔮 Prediksi Gambar"])

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'x_train' not in st.session_state:
    st.session_state.x_train = None
if 'x_test' not in st.session_state:
    st.session_state.x_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

Categories = ['GANAS', 'JINAK', 'NON KANKER']

# ==================== HALAMAN 1: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.header("Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status Data", "Belum Dimuat" if not st.session_state.data_loaded else "Sudah Dimuat")
    
    with col2:
        st.metric("Status Model", "Belum Dilatih" if st.session_state.model is None else "Sudah Dilatih")
    
    with col3:
        st.metric("Kategori", len(Categories))
    
    st.markdown("---")
    
    st.subheader("📝 Informasi Penelitian")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Judul Penelitian**: Klasifikasi Kanker dengan SVM
        
        **Metode**: Support Vector Machine (SVM)
        
        **Kategori Klasifikasi**: 
        - GANAS
        - JINAK
        - NON KANKER
        """)
    
    with col2:
        st.success("""
        **Fitur Aplikasi**:
        ✓ Loading Dataset
        ✓ Training Model
        ✓ Evaluasi Model
        ✓ Prediksi Gambar
        ✓ Confusion Matrix
        ✓ Classification Report
        """)

# ==================== HALAMAN 2: DATA LOADING ====================
elif page == "📊 Data Loading":
    st.header("Loading Dataset")
    
    current_dir = os.getcwd()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        path_ganas = os.path.join(current_dir, "Ganas")
        status_ganas = "✓ Ada" if os.path.exists(path_ganas) else "✗ Tidak Ada"
        st.write(f"📁 Folder Ganas: {status_ganas}")
    
    with col2:
        path_jinak = os.path.join(current_dir, "Jinak")
        status_jinak = "✓ Ada" if os.path.exists(path_jinak) else "✗ Tidak Ada"
        st.write(f"📁 Folder Jinak: {status_jinak}")
    
    with col3:
        path_nonkanker = os.path.join(current_dir, "Non Kanker")
        status_nonkanker = "✓ Ada" if os.path.exists(path_nonkanker) else "✗ Tidak Ada"
        st.write(f"📁 Folder Non Kanker: {status_nonkanker}")
    
    st.markdown("---")
    
    if st.button("🔄 Load Dataset", key="load_btn"):
        with st.spinner("Sedang memuat dataset..."):
            path_dict = {
                'GANAS': path_ganas,
                'JINAK': path_jinak,
                'NON KANKER': path_nonkanker
            }
            
            flat_data_arr = []
            target_arr = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_images = 0
            for category in Categories:
                folder_path = path_dict[category]
                if os.path.exists(folder_path):
                    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    total_images += len(files)
            
            loaded_images = 0
            for category in Categories:
                folder_path = path_dict[category]
                status_text.write(f"Memproses folder: {category}")
                
                if not os.path.exists(folder_path):
                    st.warning(f"Folder {category} tidak ditemukan")
                    continue
                
                files = os.listdir(folder_path)
                count = 0
                
                for img in files:
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        try:
                            img_path = os.path.join(folder_path, img)
                            img_array = imread(img_path)
                            img_resized = resize(img_array, (64, 64, 3))
                            
                            flat_data_arr.append(img_resized.flatten())
                            target_arr.append(Categories.index(category))
                            count += 1
                            loaded_images += 1
                            
                            if total_images > 0:
                                progress_bar.progress(loaded_images / total_images)
                        except Exception as e:
                            pass
                
                st.write(f"✓ {count} gambar dimuat dari {category}")
            
            # Prepare data
            flat_data = np.array(flat_data_arr)
            target = np.array(target_arr)
            
            df = pd.DataFrame(flat_data)
            df['Target'] = target
            
            # Split data
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=0.20,
                random_state=77,
                stratify=y
            )
            
            # Save to session state
            st.session_state.x_train = x_train
            st.session_state.x_test = x_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.data_loaded = True
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✓ Dataset berhasil dimuat!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Total Gambar: {len(df)}")
            with col2:
                st.info(f"Data Training: {len(x_train)} | Data Testing: {len(x_test)}")

# ==================== HALAMAN 3: TRAINING MODEL ====================
elif page == "🤖 Training Model":
    st.header("Training Model SVM")
    
    if not st.session_state.data_loaded:
        st.error("⚠️ Dataset belum dimuat. Silakan muat dataset terlebih dahulu!")
    else:
        st.info("Dataset sudah dimuat. Siap untuk training model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameter Grid Search")
            st.write("C: [1, 10, 100]")
            st.write("gamma: ['scale', 0.001]")
            st.write("kernel: ['rbf']")
        
        with col2:
            st.subheader("Konfigurasi Training")
            st.write("test_size: 0.20")
            st.write("random_state: 77")
            st.write("stratify: Yes")
        
        if st.button("▶️ Mulai Training", key="train_btn"):
            with st.spinner("Sedang melatih model... Ini mungkin memakan waktu beberapa menit"):
                import time
                start_time = time.time()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                param_grid = {
                    'C': [1, 10, 100],
                    'gamma': ['scale', 0.001],
                    'kernel': ['rbf']
                }
                
                svc = svm.SVC(probability=True)
                model = GridSearchCV(svc, param_grid, verbose=3, n_jobs=-1)
                
                status_text.write("Melakukan Grid Search...")
                progress_bar.progress(50)
                
                model.fit(st.session_state.x_train, st.session_state.y_train)
                
                progress_bar.progress(100)
                
                end_time = time.time()
                durasi = (end_time - start_time) / 60
                
                st.session_state.model = model
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("✓ Model berhasil dilatih!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"⏱️ Waktu Training: {durasi:.2f} menit")
                with col2:
                    st.info(f"🎯 Best Parameters: {model.best_params_}")

# ==================== HALAMAN 4: HASIL & EVALUASI ====================
elif page == "📈 Hasil & Evaluasi":
    st.header("Hasil & Evaluasi Model")
    
    if st.session_state.model is None:
        st.error("⚠️ Model belum dilatih. Silakan latih model terlebih dahulu!")
    else:
        st.success("✓ Model sudah dilatih. Menampilkan hasil evaluasi...")
        
        # Make predictions
        y_pred = st.session_state.model.predict(st.session_state.x_test)
        st.session_state.y_pred = y_pred
        
        # Calculate accuracy
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Akurasi", f"{accuracy*100:.2f}%")
        with col2:
            st.metric("Data Test", len(st.session_state.y_test))
        with col3:
            st.metric("Benar", int(np.sum(st.session_state.y_test == y_pred)))
        
        st.markdown("---")
        
        # Classification Report
        st.subheader("📊 Classification Report")
        report = classification_report(st.session_state.y_test, y_pred, target_names=Categories, output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrix
        st.subheader("🔲 Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=Categories,
                    yticklabels=Categories,
                    ax=ax,
                    cbar_kws={'label': 'Jumlah'})
        ax.set_xlabel('Prediksi Model')
        ax.set_ylabel('Kenyataan (Aktual)')
        ax.set_title('Confusion Matrix')
        
        st.pyplot(fig)
        
        # Explanation
        st.info("""
        **Penjelasan Confusion Matrix:**
        - Diagonal utama (warna lebih gelap) menunjukkan prediksi yang benar
        - Angka di luar diagonal menunjukkan prediksi yang salah
        - Semakin tinggi angka pada diagonal, semakin baik performa model
        """)

# ==================== HALAMAN 5: PREDIKSI GAMBAR ====================
elif page == "🔮 Prediksi Gambar":
    st.header("Prediksi Gambar")
    
    if st.session_state.model is None:
        st.error("⚠️ Model belum dilatih. Silakan latih model terlebih dahulu!")
    elif not st.session_state.data_loaded:
        st.error("⚠️ Dataset belum dimuat.")
    else:
        st.success("✓ Model siap digunakan untuk prediksi.")
        
        st.markdown("---")
        
        # Pilihan prediksi
        option = st.radio("Pilih cara prediksi:", 
                         ["Dari Data Test", "Upload Gambar"])
        
        if option == "Dari Data Test":
            st.subheader("Prediksi dari Data Test")
            
            num_samples = st.slider("Jumlah sampel yang ditampilkan:", 1, 9, 6)
            
            if st.button("🎲 Tampilkan Prediksi", key="predict_btn"):
                jumlah_sample = min(num_samples, len(st.session_state.x_test))
                
                fig, axes = plt.subplots(2, (jumlah_sample + 1) // 2, figsize=(15, 8))
                axes = axes.flatten()
                
                for i in range(jumlah_sample):
                    img_data = st.session_state.x_test.iloc[i].values.reshape(64, 64, 3)
                    
                    probabilitas = st.session_state.model.predict_proba([st.session_state.x_test.iloc[i]])[0]
                    
                    prediksi_idx = st.session_state.y_pred[i]
                    prediksi_lbl = Categories[prediksi_idx]
                    aktual_lbl = Categories[int(st.session_state.y_test.iloc[i])]
                    confidence = probabilitas[prediksi_idx] * 100
                    
                    ax = axes[i]
                    ax.imshow(img_data)
                    ax.axis('off')
                    
                    warna = 'green' if prediksi_lbl == aktual_lbl else 'red'
                    
                    ax.set_title(f"Pred: {prediksi_lbl}\n({confidence:.1f}%)\nAsli: {aktual_lbl}",
                               color=warna, fontsize=10, fontweight='bold')
                
                # Hide extra subplots
                for j in range(jumlah_sample, len(axes)):
                    axes[j].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        else:  # Upload Gambar
            st.subheader("Upload Gambar untuk Prediksi")
            
            uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png', 'bmp'])
            
            if uploaded_file is not None:
                image = imread(uploaded_file)
                
                # Preprocess
                img_resized = resize(image, (64, 64, 3))
                img_flat = img_resized.flatten().reshape(1, -1)
                
                # Predict
                pred_idx = st.session_state.model.predict(img_flat)[0]
                pred_label = Categories[pred_idx]
                probabilitas = st.session_state.model.predict_proba(img_flat)[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Hasil Prediksi")
                    st.metric("Prediksi", pred_label, help="Kategori yang diprediksi")
                    st.metric("Confidence", f"{probabilitas[pred_idx]*100:.2f}%")
                    
                    st.markdown("---")
                    st.subheader("Probabilitas Semua Kategori")
                    
                    prob_data = {
                        'Kategori': Categories,
                        'Probabilitas': probabilitas,
                        'Persen': [f"{p*100:.2f}%" for p in probabilitas]
                    }
                    prob_df = pd.DataFrame(prob_data)
                    st.dataframe(prob_df, use_container_width=True)
                    
                    # Bar chart
                    fig, ax = plt.subplots()
                    colors = ['green' if i == pred_idx else 'lightblue' for i in range(len(Categories))]
                    ax.bar(Categories, probabilitas, color=colors)
                    ax.set_ylabel('Probabilitas')
                    ax.set_title('Probabilitas Prediksi')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 Aplikasi Klasifikasi Kanker dengan SVM</p>", unsafe_allow_html=True)
