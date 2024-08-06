import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import cv2
import dlib

def load_model():
    model_path = 'streamlit_app/model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

model = load_model()

detector = dlib.get_frontal_face_detector()

def detect_and_extract_face(image):
    img_array = np.array(image)
    faces = detector(img_array, 1)
    for i, d in enumerate(faces):
        x, y, w, h = d.left(), d.top(), d.width(), d.height()
        roi_color = img_array[y : y + h, x : x + w]
        face_img = Image.fromarray(roi_color).resize((128, 128))
        return face_img, (x, y, w, h)
    return None, None

def classify_liveness(face_img):
    face_array = np.array(face_img) / 255.0
    face_tensor = np.expand_dims(face_array, axis=0)
    prediction = model.predict(face_tensor)
    return prediction[0][0]

st.title('Cognitive Environments - Detecção de Vivacidade')

option = st.radio("Escolha uma opção:", ('Carregar Imagem', 'Capturar pela Webcam'))

if option == 'Carregar Imagem':
    uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")
else:
    camera = st.camera_input("Tire uma foto")

if uploaded_file or camera:
    if uploaded_file is not None:
        img_stream = io.BytesIO(uploaded_file.getvalue())
        image = Image.open(img_stream).convert("RGB")
    elif camera is not None:
        bytes_data = camera.getvalue()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

    with st.spinner("Classificando imagem..."):
        face_img, bbox = detect_and_extract_face(image)
        if face_img is not None:
            prediction = classify_liveness(face_img)
            if prediction >= 0.5:
                result = "Vivo"
                color = (0, 255, 0)
            else:
                result = "Fraudulento"
                color = (255, 0, 0)

            x, y, w, h = bbox
            image_np = np.array(image)
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)

            st.image(image_np, channels="RGB")
            st.success(f"Classificação: {result}, Pontuação de vivacidade: {prediction * 100:.2f}%")
        else:
            st.error("Nenhum rosto detectado. Tente novamente com uma imagem diferente.")