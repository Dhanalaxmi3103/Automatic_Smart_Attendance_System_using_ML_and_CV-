# Automatic_Smart_Attendance_System_using_ML_and_CV
# 🎓 Automatic Smart Attendance System using Machine Learning and Computer Vision

## 📖 Overview
This project is a Smart Attendance System that automates attendance tracking using face recognition.  
It leverages Machine Learning algorithms (HOG, PCA, SVM) and Computer Vision (Haar Cascade) to detect and recognize faces in real-time using a webcam.  
Attendance is automatically marked in a database, eliminating manual entry and preventing proxy attendance.

---

## ⚙️ Features
- Real-time face detection and recognition via webcam
- Prevents proxy attendance and ensures identity verification
- Automatically updates attendance in a digital database
- User-friendly Streamlit interface for registration and tracking
- Supports addition of new students via live capture or image upload
- Works under different lighting and pose conditions

---

## 🧠 Algorithms and Techniques
- Haar Cascade – for face detection  
- HOG (Histogram of Oriented Gradients) – for feature extraction  
- PCA (Principal Component Analysis)– for dimensionality reduction  
- SVM (Support Vector Machine) – for classification and recognition  

---

## 🗂️ Project Structure
Automatic_Smart_Attendance_System_using_ML_and_CV/
│
├── app.py # Main Streamlit app
├── svm_model_pca.pkl # Trained SVM model
├── scaler.pkl # Feature scaler
├── pca_transform.pkl # PCA transformer
├── label_mapping1.txt # Label mapping file
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── requirements.txt # Required dependencies
└── README.md # Project documentation

## 🚀 Deployment (Streamlit Cloud)
1. Push all the above files to a **public GitHub repository**.  
2. Go to [https://share.streamlit.io](https://share.streamlit.io).  
3. Sign in with GitHub and select your repository.  
4. Choose the main branch and set the entry file as:
5. 5. Click **Deploy** – Streamlit will build and host your app online.

---

## 🧰 Libraries Used
- `streamlit` – for building web interface  
- `opencv-python` – for image capture and face detection  
- `numpy` – for numerical computation  
- `pandas` – for data handling  
- `scikit-learn` – for ML algorithms (SVM, PCA)  
- `joblib` – for model saving/loading  
- `datetime` – for marking attendance time  
 
## 👩‍💻 Author
Dhanalaxmi3103
Smart Attendance System using Machine Learning and Computer Vision

## 🏁 Conclusion
This system provides a fast, accurate, and automated attendance solution that minimizes human error and enhances efficiency for educational and corporate institutions.

