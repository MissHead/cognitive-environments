import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import io
from PIL import Image
import subprocess
import sys
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = '/Users/bela/Documents/GitHub/cognitive-environments/streamlit_app/model.h5'
liveness_model = load_model(model_path)

# Carregar o detector de faces dlib
detector = dlib.get_frontal_face_detector()

# Função para detectar e extrair o rosto
def detect_and_extract_face(image):
    img_array = np.array(image)
    faces = detector(img_array, 1)
    for i, d in enumerate(faces):
        x, y, w, h = d.left(), d.top(), d.width(), d.height()
        roi_color = img_array[y : y + h, x : x + w]
        face_img = Image.fromarray(roi_color).resize((128, 128))
        return face_img, (x, y, w, h)
    return None, None

# Função para classificar a vivacidade
def classify_liveness(face_img):
    face_array = np.array(face_img) / 255.0
    face_tensor = np.expand_dims(face_array, axis=0)  # Keep the shape as (1, 128, 128, 3)
    print("Shape of face tensor before passing to model:", face_tensor.shape)  # Debugging
    prediction = liveness_model.predict(face_tensor)
    return prediction[0][0]

# Streamlit interface
st.title("| Cognitive Environments | FIAP | 2024 |")
st.header("| Izabela Ramos Ferreira      | RM 352447")
st.header("| Kaique Vinicius Lima Soares | RM 351437")
st.header("| Walder Octacilio Garbellott | RM 352469")


uploaded_file = st.file_uploader("Tente uma outra imagem", type=["png", "jpg"])
camera = st.camera_input(
    "Tire sua foto",
    help="Lembre-se de permitir ao seu navegador o acesso a sua câmera.",
)

if camera or uploaded_file:
    if uploaded_file is not None:
        img_stream = io.BytesIO(uploaded_file.getvalue())
        imagem = Image.open(img_stream).convert("RGB")
    elif camera is not None:
        bytes_data = camera.getvalue()
        imagem = Image.open(io.BytesIO(bytes_data)).convert("RGB")

    with st.spinner("Classificando imagem..."):
        face_img, bbox = detect_and_extract_face(imagem)
        if face_img is not None:
            prediction = classify_liveness(face_img)
            if prediction >= 0.5:
                result = "Rosto Real"
                color = (0, 255, 0)
            else:
                result = "Rosto Forjado"
                color = (0, 0, 255)

            # Desenhar a bounding box na imagem original
            x, y, w, h = bbox
            imagem_np = np.array(imagem)
            cv2.rectangle(imagem_np, (x, y), (x + w, y + h), color, 2)

            st.image(imagem_np, channels="RGB")
            st.success(
                f"Classificação: {result}, Pontuação de vivacidade: {prediction * 100:.2f}%"
            )
        else:
            st.error(
                "Nenhum rosto detectado. Tente novamente com uma imagem diferente."
            )