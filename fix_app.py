with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the problematic section
old_text = """        st.markdown(\"\"\"
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### <i class='fas fa-cogs'></i> Metodologi & Technology Stack", unsafe_allow_html=True)
        
        - **Algorithm**: Support Vector Machine (SVM) with Linear Kernel
        - **Image Processing**: scikit-image
        - **Framework**: Streamlit
        - **Visualization**: Plotly, Matplotlib
        - **Model Training**: scikit-learn
        
        ## � Model Specifications
        
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
        
        ## 🚀 Cara Penggunaan Sistem
        \"\"\")
        
        with st.container():"""

new_text = """        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### <i class='fas fa-cogs'></i> Metodologi & Technology Stack", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container():"""

if old_text in content:
    content = content.replace(old_text, new_text)
    print("✓ Fixed Metodologi section")
else:
    # Try alternate approach - look for unique markers
    import re
    pattern = r'st\.markdown\("\"\"\s+st\.markdown\("<br>.*?with st\.container\(\):'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + new_text + content[match.end():]
        print("✓ Fixed using pattern matching")
    else:
        print("✗ Could not find the problematic section")

# Replace all emoticon numbers
content = content.replace('**1️⃣', '**1.')
content = content.replace('**2️⃣', '**2.')
content = content.replace('**3️⃣', '**3.')
content = content.replace('**4️⃣', '**4.')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
    
print("✓ File fixed and saved")
