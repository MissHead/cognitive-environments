import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

@st.cache_resource
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

if option == 'Carregar Imagem':
    uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")
else:
    uploaded_file = capture_image()

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem carregada.', use_column_width=True)
        
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        st.write("Tamanho da imagem:", image_array.shape)
        
        prediction = model.predict(image_array, batch_size=1)[0][0]
        
        result = 'Vivo' if prediction > 0.5 else 'Fraudulento'
        st.write(f"Resultado da detecção: {result}")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a predição: {e}")