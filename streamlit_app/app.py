import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import io

model = tf.keras.models.load_model('streamlit_app/model.h5')

st.title('Detecção de Vivacidade')

def capture_image():
    picture = st.camera_input("Tire uma foto")
    return picture

# Opção para carregar imagem ou capturar pela webcam
option = st.radio("Escolha uma opção:", ('Carregar Imagem', 'Capturar pela Webcam'))

if option == 'Carregar Imagem':
    uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")
else:
    uploaded_file = capture_image()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem carregada.', use_column_width=True)
    
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)[0][0]
    
    result = 'Vivo' if prediction > 0.5 else 'Fraudulento'
    st.write(f"Resultado da detecção: {result}")