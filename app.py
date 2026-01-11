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
    # Load model
    model, scaler, error = load_model_and_scaler()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("💡 Please ensure the model files are available.")
        st.stop()
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital-3.png", width=80)
        st.markdown("## 🏥 Cancer Classifier")
        st.markdown("---")
        
        page = st.radio(
            "📋 Navigation",
            ["🏠 Dashboard", "🔍 Image Classification", "📊 Batch Processing", "📈 Model Evaluation"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Classification Types")
        st.markdown("""
        - 🔴 **GANAS** - Malignant
        - 🟡 **JINAK** - Benign
        - 🟢 **NON KANKER** - Healthy
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown("""
        **Algorithm**: SVM Linear  
        **Resolution**: 224×224×3  
        **Features**: 150,528  
        **Classes**: 3 Types
        """)
    
    # ==================== PAGE 1: DASHBOARD ====================
    if page == "🏠 Dashboard":
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🏥 Cancer Classification System</h1>
            <p class="header-subtitle">Advanced AI-Powered Medical Image Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 👋 Welcome to Cancer Classification System")
        st.markdown("""
        This system uses **Support Vector Machine (SVM)** with linear kernel 
        to classify histopathology images into three categories.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="result-card">
                <h3 style="color: #dc3545;">🔴 GANAS</h3>
                <p><strong>Malignant Cancer</strong></p>
                <p>Cancerous tumors that can spread to other parts of the body.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="result-card">
                <h3 style="color: #ffc107;">🟡 JINAK</h3>
                <p><strong>Benign Tumor</strong></p>
                <p>Non-cancerous tumors that do not spread.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="result-card">
                <h3 style="color: #28a745;">🟢 NON KANKER</h3>
                <p><strong>Healthy Tissue</strong></p>
                <p>Normal, healthy tissue with no abnormalities.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🚀 Quick Start Guide")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Single Image Classification:**
            1. Go to "Image Classification" page
            2. Upload a medical image (JPG/PNG)
            3. Click "Analyze Image"
            4. View prediction & confidence scores
            """)
        
        with col2:
            st.markdown("""
            **📊 Batch Processing:**
            1. Go to "Batch Processing" page
            2. Upload multiple images at once
            3. Click "Process All Images"
            4. Download results as CSV file
            """)
        
        st.markdown("---")
        
        st.markdown("### 🔬 Technology Stack")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Machine Learning:**
            - Algorithm: SVM with Linear Kernel
            - Training: scikit-learn
            - Image Processing: scikit-image
            - Data Augmentation: 4x factor
            """)
        
        with col2:
            st.markdown("""
            **Preprocessing:**
            - Input: 224×224×3 RGB images
            - Normalization: [0, 1] range
            - Scaling: StandardScaler
            - Auto grayscale conversion
            """)
        
        st.markdown("---")
        
        st.warning("""
        ⚠️ **Disclaimer**: This system is for research and educational purposes only. 
        Always consult with qualified medical professionals for proper diagnosis and treatment.
        """)
    
    # ==================== PAGE 2: IMAGE CLASSIFICATION ====================
    elif page == "🔍 Image Classification":
    # ==================== PAGE 2: IMAGE CLASSIFICATION ====================
    elif page == "🔍 Image Classification":
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🔍 Image Classification</h1>
            <p class="header-subtitle">Upload and analyze single medical image</p>
        </div>
        """, unsafe_allow_html=True)
        
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
    
    # ==================== PAGE 3: BATCH PROCESSING ====================
    elif page == "📊 Batch Processing":
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">📊 Batch Processing</h1>
            <p class="header-subtitle">Process multiple images at once</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Upload Multiple Images")
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
    
    # ==================== PAGE 4: MODEL EVALUATION ====================
    elif page == "📈 Model Evaluation":
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">📈 Model Evaluation</h1>
            <p class="header-subtitle">Training Results & Performance Metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load model info if exists
        info_files = [f for f in os.listdir('.') if f.startswith('cancer_model_info_') and f.endswith('.txt')]
        
        if info_files:
            info_file = sorted(info_files)[-1]
            
            st.markdown("### 📄 Model Training Information")
            
            with open(info_file, 'r') as f:
                model_info = f.read()
            
            # Parse info file
            lines = model_info.split('\n')
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            for line in lines:
                if 'Training Accuracy' in line:
                    acc = line.split(':')[1].strip()
                    with col1:
                        st.metric("🎯 Training Accuracy", acc)
                elif 'Testing Accuracy' in line:
                    test_acc = line.split(':')[1].strip()
                    with col2:
                        st.metric("✅ Testing Accuracy", test_acc)
                elif 'Total Images' in line:
                    total = line.split(':')[1].strip()
                    with col3:
                        st.metric("📊 Total Images", total)
                elif 'Training Samples' in line:
                    train_samples = line.split(':')[1].strip()
                    with col4:
                        st.metric("🔢 Training Samples", train_samples)
            
            st.markdown("---")
            
            # Display full info
            with st.expander("📋 View Complete Training Report"):
                st.code(model_info)
            
            st.markdown("---")
            
            # Model specifications
            st.markdown("### 🔬 Model Specifications")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Architecture:**
                - Algorithm: Support Vector Machine (SVM)
                - Kernel: Linear
                - Probability Estimates: Enabled
                - Class Weight: Balanced
                
                **Input Processing:**
                - Image Size: 224 × 224 pixels
                - Color Channels: 3 (RGB)
                - Total Features: 150,528
                - Normalization: [0, 1] range
                """)
            
            with col2:
                st.markdown("""
                **Training Configuration:**
                - Data Split: 80% Train / 20% Test
                - Random State: 77
                - Stratified Sampling: Yes
                - Augmentation Factor: 4x
                
                **Augmentation Techniques:**
                - Rotation: ±30 degrees
                - Horizontal Flip
                - Vertical Flip
                - Brightness Adjustment: ±20%
                """)
            
            st.markdown("---")
            
            # Classification categories
            st.markdown("### 🎯 Classification Categories")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style="background: #dc3545; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h2>🔴 GANAS</h2>
                    <p><strong>Malignant Cancer</strong></p>
                    <p>Cancerous tumors that can spread and invade other tissues</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: #ffc107; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h2>🟡 JINAK</h2>
                    <p><strong>Benign Tumor</strong></p>
                    <p>Non-cancerous tumors that do not spread</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style="background: #28a745; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h2>🟢 NON KANKER</h2>
                    <p><strong>Healthy Tissue</strong></p>
                    <p>Normal tissue with no abnormalities</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Performance notes
            st.markdown("### 📊 Performance Analysis")
            
            st.info("""
            **Model Performance Highlights:**
            - The model was trained on histopathology images with 4x data augmentation
            - Standard scaling was applied to normalize feature values
            - Class balancing ensures fair treatment of all categories
            - Testing accuracy represents performance on unseen data
            """)
            
            st.success("""
            **Strengths:**
            - High-resolution input (224×224) captures fine morphological details
            - Data augmentation improves model generalization
            - Linear kernel SVM provides fast inference speed
            - Probability estimates enable confidence scoring
            """)
            
            st.warning("""
            **Limitations:**
            - Performance depends on image quality and proper lighting
            - Small dataset may limit generalization to diverse cases
            - Not a replacement for professional medical diagnosis
            - Should be used as a screening tool only
            """)
            
        else:
            st.warning("⚠️ Model training information file not found.")
            st.info("💡 The model info file should be generated during training in the Jupyter notebook.")
            
            # Show sample metrics format
            st.markdown("### 📊 Expected Metrics")
            st.markdown("""
            When the model is trained, you'll see:
            - Training & Testing Accuracy
            - Confusion Matrix
            - Classification Report (Precision, Recall, F1-Score)
            - Per-class Performance Metrics
            """)

# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    main()