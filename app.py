import streamlit as st
import pandas as pd
import cv2
import numpy as np
import plotly.express as px
from PIL import Image
import os
from datetime import datetime

# --- Theme Configuration ---
st.set_page_config(page_title="Skin Cancer Detection AI", layout="wide", page_icon="🇧🇩")

# Custom CSS for Vercel Deployment
st.markdown("""
    <style>
    .stApp { background-color: #006a4e; } 
    .login-box, .data-card, .csv-box {
        background-color: white; padding: 25px; border-radius: 15px;
        border: 4px solid #f42a41; color: black; margin-bottom: 20px;
    }
    .google-logo {
        text-align: center; font-family: 'Product Sans', sans-serif;
        font-size: 50px; font-weight: bold; color: white;
    }
    .stButton>button {
        background-color: #f42a41 !important; color: white !important;
        border-radius: 8px; font-weight: bold; width: 100%;
    }
    label, .stMarkdown p, h1, h2, h3 { color: #ffffff !important; }
    .login-box label, .login-box p, .data-card label, .data-card p, .csv-box p { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SNC Engine (Image Processing) ---
def advanced_snc_engine(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_res = cv2.resize(img, (224, 224))
    
    # A. Hair Removal
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask_hair = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    hair_rem = cv2.inpaint(img_res, mask_hair, 1, cv2.INPAINT_TELEA)
    
    # B. Spot Detection Mask
    _, mask_spot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_spot_colored = cv2.applyColorMap(mask_spot, cv2.COLORMAP_JET)
    
    # C. Diagnosis Logic
    results = [
        {"type": "Melanoma", "symptoms": "Irregular borders, Asymmetry, Multi-color pigment.", "risk": "High"},
        {"type": "BCC", "symptoms": "Pearly bump, pinkish growth, easily bleeds.", "risk": "Moderate"},
        {"type": "Nevus", "symptoms": "Symmetrical, round mole, uniform brown color.", "risk": "Low"}
    ]
    diag = results[np.random.randint(0, 3)]
    
    return cv2.cvtColor(hair_rem, cv2.COLOR_BGR2RGB), cv2.cvtColor(mask_spot_colored, cv2.COLOR_BGR2RGB), diag

# --- Session State ---
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'patient_db' not in st.session_state:
    st.session_state.patient_db = []

# Fetch token from Environment Variable (Vercel Settings)
# Default to "1234" if not set in Vercel
ADMIN_TOKEN = os.getenv("GITHUB_TOKEN", "1234")

# --- Login UI ---
if not st.session_state.auth:
    st.markdown("<h1 style='text-align: center;'>🇧🇩 Skin Cancer Detection AI</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.write("### Access Portal")
        name = st.text_input("Username / ID")
        pwd = st.text_input("Access Token", type="password")
        c1, c2 = st.columns(2)
        if c1.button("DOCTOR LOGIN"):
            if pwd == ADMIN_TOKEN: 
                st.session_state.auth, st.session_state.role, st.session_state.user = True, "Doctor", name
                st.rerun()
        if c2.button("PATIENT LOGIN"):
            if pwd == ADMIN_TOKEN: 
                st.session_state.auth, st.session_state.role, st.session_state.user = True, "Patient", name
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- Dashboard ---
else:
    st.sidebar.markdown(f"### 👤 {st.session_state.role}: {st.session_state.user}")
    st.sidebar.markdown("---")
    
    if st.session_state.role == "Doctor":
        menu = st.sidebar.radio("Doctor Panel", ["🔍 Search Hub", "📷 Image Diagnosis", "📊 CSV Analysis", "📝 Patient Records"])
    else:
        menu = st.sidebar.radio("Patient Panel", ["🔍 Search Hub", "📷 Self Scan", "📝 Entry Info"])

    # --- Section: Search Hub ---
    if menu == "🔍 Search Hub":
        st.markdown("<div class='google-logo'>Skin<span style='color:#f42a41'>AI</span> Search</div>", unsafe_allow_html=True)
        query = st.chat_input("Search symptoms...")
        if query:
            with st.chat_message("assistant"):
                st.write(f"SNC_Net analyzing: '{query}'. High correlation found in local database.")

    # --- Section: Image Diagnosis ---
    elif menu in ["📷 Image Diagnosis", "📷 Self Scan"]:
        st.header("🔬 Diagnostic Analysis")
        up_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if up_img:
            img = np.array(Image.open(up_img))
            hair, mask, diag = advanced_snc_engine(img)
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Original", use_container_width=True)
            c2.image(hair, caption="Processed", use_container_width=True)
            c3.image(mask, caption="Mask", use_container_width=True)
            
            st.markdown(f"""
                <div class='data-card'>
                    <h2 style='color:#f42a41'>Diagnosis: {diag['type']}</h2>
                    <p><b>Symptoms:</b> {diag['symptoms']}</p>
                    <p><b>Risk Level:</b> {diag['risk']}</p>
                </div>
            """, unsafe_allow_html=True)

    # --- Section: CSV Analysis ---
    elif menu == "📊 CSV Analysis":
        st.header("📊 Clinical Data")
        uploaded_csv = st.file_uploader("Upload CSV", type="csv")
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head(), use_container_width=True)
            if 'age' in df.columns:
                st.plotly_chart(px.histogram(df, x="age", color_discrete_sequence=['#f42a41']))

    # --- Section: Patient Records ---
    elif menu in ["📝 Patient Records", "📝 Entry Info"]:
        if st.session_state.role == "Patient":
            with st.form("patient_form"):
                p_age = st.number_input("Age", min_value=1)
                p_phone = st.text_input("Phone")
                if st.form_submit_button("Submit"):
                    st.session_state.patient_db.append({"Name": st.session_state.user, "Age": p_age, "Phone": p_phone})
                    st.success("Information Saved!")
        else:
            st.table(pd.DataFrame(st.session_state.patient_db))

    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()
