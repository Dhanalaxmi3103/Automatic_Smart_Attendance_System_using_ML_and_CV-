import streamlit as st
import cv2
import numpy as np
import os
import joblib
import pandas as pd
import sqlite3
from datetime import datetime
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Set up layout
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("🎓 Smart Attendance System")
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["📸 Register Student", "🤖 Automate Attendance", "📋 View Attendance", "✅ Check Registered Students"])

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Initialize attendance database
def initialize_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
        conn.commit()
    conn.close()

# Extract features and retrain model
def extract_features_and_train(dataset_path="dataset"):
    X, y = [], []
    label_map = {}
    current_label = 0

    for person in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person)
        if not os.path.isdir(person_folder): continue

        label_map[current_label] = person
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            face_feat = hog(cv2.resize(img, (128, 128)), orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
            X.append(face_feat)
            y.append(current_label)
        current_label += 1

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_pca, y)

    joblib.dump(model, "svm_model_pca3.pkl")
    joblib.dump(scaler, "scaler3.pkl")
    joblib.dump(pca, "pca_transform3.pkl")

    with open("label_mapping1.txt", "w") as f:
        for k, v in label_map.items():
            f.write(f"{k},{v}\n")

    return len(label_map)

# Load trained model
def load_model():
    model = joblib.load("svm_model_pca3.pkl")
    scaler = joblib.load("scaler3.pkl")
    pca = joblib.load("pca_transform3.pkl")
    label_map = {}
    with open("label_mapping1.txt", "r") as f:
        for line in f:
            idx, name = line.strip().split(",", maxsplit=1)
            label_map[int(idx)] = name.strip()
    return model, scaler, pca, label_map

# === 📸 Register New Student ===
if menu == "📸 Register Student":
    st.subheader("➕ Register New Student")
    name = st.text_input("Enter Student Name")
    sid = st.text_input("Enter Student ID")
    capture_type = st.radio("Choose input type:", ["Live Webcam", "Manual Upload"])
    num_images = st.slider("Number of images to capture/upload", 5, 50, 10)

    folder = f"dataset/{name}_{sid}"
    os.makedirs(folder, exist_ok=True)

    if capture_type == "Live Webcam":
        if st.button("Start Capture") and name and sid:
            cap = cv2.VideoCapture(0)
            count = 0
            stframe = st.empty()
            while count < num_images:
                ret, frame = cap.read()
                if not ret:
                    break
                faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
                for (x, y, w, h) in faces:
                    roi = frame[y:y+h, x:x+w]
                    cv2.imwrite(f"{folder}/{count}.jpg", roi)
                    count += 1
                    break
                stframe.image(frame, channels="BGR")
            cap.release()
            register_student(name, sid)
            st.success(f"Student {name} registered with webcam images")

    else:  # Manual Upload
        uploaded_files = st.file_uploader("Upload images", type=["jpg", "png"], accept_multiple_files=True)
        if uploaded_files and name and sid:
            if st.button("Upload Images"):
                for i, file in enumerate(uploaded_files[:num_images]):
                    img = Image.open(file)
                    img.save(f"{folder}/{i}.jpg")
                register_student(name, sid)
                st.success(f"Student {name} registered with uploaded images")

# === 🤖 Automate Attendance ===
elif menu == "🤖 Automate Attendance":
    st.subheader("📷 Real-time Automated Attendance")
    initialize_database()

    if 'start_webcam' not in st.session_state:
        st.session_state.start_webcam = False

    if not st.session_state.start_webcam:
        if st.button("▶️ Start Webcam"):
            st.session_state.start_webcam = True
            st.rerun()

    if st.session_state.start_webcam:
        if st.button("⏹ Stop Webcam"):
            if st.checkbox("✅ Confirm stop webcam"):
                st.session_state.start_webcam = False
                st.rerun()

        model, scaler, pca, label_map = load_model()
        expected_length = scaler.mean_.shape[0]

        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

        while st.session_state.start_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam Error"); break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (128, 128))

                face_feat = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                combined_feat = np.array(face_feat)

                if len(combined_feat) < expected_length:
                    combined_feat = np.pad(combined_feat, (0, expected_length - len(combined_feat)), mode='constant')
                elif len(combined_feat) > expected_length:
                    combined_feat = combined_feat[:expected_length]

                combined_feat_scaled = scaler.transform([combined_feat])
                combined_feat_pca = pca.transform(combined_feat_scaled)
                pred_idx = model.predict(combined_feat_pca)[0]
                name = label_map.get(pred_idx, "Unknown")

                if name != "Unknown":
                    mark_attendance(name)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        st.success("🛑 Webcam stopped.")

# === 📋 View Attendance ===
elif menu == "📋 View Attendance":
    st.subheader("📋 View Attendance Records")

    try:
        conn = sqlite3.connect("attendance.db")
        df = pd.read_sql_query("SELECT * FROM attendance ORDER BY date DESC, time DESC", conn)
        conn.close()

        if not df.empty:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download as CSV", data=csv, file_name='attendance_records.csv', mime='text/csv')
        else:
            st.info("No attendance records found.")
    except Exception as e:
        st.error(f"Failed to load attendance records: {e}")

    date = st.date_input("Select date")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE date=?", (date.strftime("%Y-%m-%d"),))
    data = c.fetchall()
    conn.close()
    if data:
        st.write("### Attendance Records")
        st.dataframe(data, use_container_width=True)
    else:
        st.info("No attendance records found for this date.")

# === ✅ Check Registered Students ===
elif menu == "✅ Check Registered Students":
    st.subheader("✅ Registered Students")
    try:
        with open("label_mapping1.txt", "r") as f:
            labels = f.readlines()
        if labels:
            st.write("Total Registered:", len(labels))
            for line in labels:
                idx, name = line.strip().split(",", 1)
                st.write(f"{idx}. {name}")
        else:
            st.info("No registered students found.")
    except Exception as e:
        st.error(f"Could not load student list: {e}")


