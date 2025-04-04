
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="Punch Out Pro - AI Grout Detector", layout="wide")
st.title("Punch Out Pro - AI Grout Detector")

uploaded_file = st.file_uploader("Upload a walkthrough video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    model = YOLO("yolov8n.pt")  # You can replace this with a custom model path

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Missing Grout", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        stframe.image(frame, channels="BGR")
    cap.release()
