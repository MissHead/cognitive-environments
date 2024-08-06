import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = 'streamlit_app/model.h5'
model = tf.keras.models.load_model(model_path)

st.title('Cognitive Environments - Detecção de Vivacidade')

def capture_image():
    picture = st.camera_input("Tire uma foto")
    return picture

option = st.radio("Escolha uma opção:", ('Carregar Imagem', 'Capturar pela Webcam'))

uploaded_file = None
camera = None

if option == 'Carregar Imagem':
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg"])
else:
    camera = capture_image()

def process_image(image_file):
    try:
        if image_file is not None:
            img_stream = io.BytesIO(image_file.getvalue())
            image = Image.open(img_stream).convert("RGB")
            return image
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
    return None

image = None

if uploaded_file:
    image = process_image(uploaded_file)
elif camera:
    image = process_image(camera)

if image:
    with st.spinner("Classificando imagem..."):
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        st.write("Tamanho da imagem:", image_array.shape)

        prediction = model.predict(image_array, batch_size=1)[0][0]

        if prediction is not None:
            result = 'Vivo' if prediction > 0.5 else 'Fraudulento'
            st.image(image, caption=f"Classificação: {result}, Pontuação de vivacidade: {prediction * 100:.2f}%")
        else:
            st.error("A predição falhou.")
else:
    st.info("Por favor, carregue uma imagem ou capture uma foto usando a webcam.")