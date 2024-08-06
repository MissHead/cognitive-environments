import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import dlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = 'streamlit_app/xmodel.h5'
xmodel = tf.keras.models.load_model(model_path)


detector = dlib.get_frontal_face_detector()


def detect_and_extract_face(image):
    img_array = np.array(image)
    faces = detector(img_array, 1)
    for i, d in enumerate(faces):
        x, y, w, h = d.left(), d.top(), d.width(), d.height()
        roi_color = img_array[y : y + h, x : x + w]
        face_img = Image.fromarray(roi_color).resize((224, 224))
        return face_img, (x, y, w, h)
    return None, None


def classify_liveness(face_img):
    face_array = np.array(face_img) / 255.0
    face_tensor = np.expand_dims(face_array, axis=0)
    prediction = xmodel.predict(face_tensor)
    return prediction[0][0]


st.title('Cognitive Environments - Detecção de Vivacidade')
st.header("| Izabela Ramos Ferreira      | RM 352447")
st.header("| Kaique Vinicius Lima Soares | RM 351437")
st.header("| Walder Octacilio Garbellott | RM 352469")
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

        face_img, bbox = detect_and_extract_face(image)
        if face_img is not None:
            prediction = classify_liveness(face_img)
            if prediction >= 0.5:
                result = "Rosto Real"
                color = (0, 255, 0)
            else:
                result = "Rosto Forjado"
                color = (0, 0, 255)

            x, y, w, h = bbox
            imagem_np = np.array(image)

            st.image(imagem_np, channels="RGB")
            st.success(
                f"Classificação: {result}, Pontuação de vivacidade: {prediction * 100:.2f}%"
            )
        else:
            st.error(
                "Nenhum rosto detectado. Tente novamente com uma imagem diferente."
            )
else:
    st.info("Por favor, carregue uma imagem ou capture uma foto usando a webcam.")