
# 🇧🇩 Skin Cancer Detection AI
#live
A specialized diagnostic portal for early-stage skin cancer screening, powered by a hybrid Deep Learning architecture .
## 🌐 Live Demo
You can access the live web application here:
**[Skin Cancer Detection AI - Live Portal](https://skin-cancer-detection-ai-ayysneeq2vsbaphwlhhkb7.streamlit.app/)**
 --login   1. doctor _username : Admin _ password: 1234  
 __login   2 .patient_username : patient_password:1234 

## 🚀 Key Features
- **SNC_Net Engine:** Hybrid feature extraction with a stacking ensemble approach.
- **High Performance:** Achieved a validation accuracy of **97.81%**.
- **Dual Portal Access:** Separate interfaces for **Doctors** (Batch Analytics) and **Patients** (Self-scan).
- **Clinical Pre-processing:** Automated hair removal (DullRazor) and lesion spot masking.
- **Skin AI Search Hub:** Integrated AI chatbot for disease inquiries on local IP.

## 📊 Methodology Highlights
- **Artifact Removal:** Morphological Black-hat transformation and Inpainting.
- **Segmentation:** Otsu's adaptive thresholding for precise lesion isolation.
- **Classification:** Categorization into 7 classes (Melanoma, BCC, Nevus, etc.).

## 🛠️ Built With
- **Framework:** [Streamlit]([https://streamlit.io/](https://skin-cancer-detection-ai-ayysneeq2vsbaphwlhhkb7.streamlit.app/))
- **Image Processing:** OpenCV, PIL
- **Data Visualization:** Plotly Express, Pandas
- **Backend:** Python 3.10+

## 📂 Project Structure
- `app.py`: The main application script.
 `db_skin.sql`: The database script.
- `requirements.txt`: List of Python dependencies.
- `images/`: UI assets and sample lesion data.
- `models/` (Optional): Pre-trained weights for SNC_Net.

---
*Developed as part of an Undergraduate Thesis Project.*
