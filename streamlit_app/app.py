import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

def load_model():
    model_path = 'streamlit_app/model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

model = load_model()

st.title('Cognitive Environments - Detecção de Vivacidade')

def capture_image():
    picture = st.camera_input("Tire uma foto")
    return picture

option = st.radio("Escolha uma opção:", ('Carregar Imagem', 'Capturar pela Webcam'))

uploaded_file = None
camera = None

if option == 'Carregar Imagem':
    uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")
else:
    camera = capture_image()

if uploaded_file or camera:
    if uploaded_file is not None:
        img_stream = io.BytesIO(uploaded_file.getvalue())
        image = Image.open(img_stream).convert("RGB")
    elif camera is not None:
        bytes_data = camera.getvalue()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

    with st.spinner("Classificando imagem..."):
        try:
            image = image.resize((128, 128))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array, batch_size=1)[0][0]

            result = 'Vivo' if prediction > 0.5 else 'Fraudulento'
            st.image(image, caption=f"Classificação: {result}, Pontuação de vivacidade: {prediction * 100:.2f}%")
        except Exception as e:
            st.error(f"Ocorreu um erro durante a predição: {e}")