import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import mtcnn

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Deepfake",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO PARA DEIXAR MAIS BONITO ---
# Este bloco de CSS vai criar o estilo dos 'cards' e esconder o rodapé do Streamlit
st.markdown("""
<style>
/* Estilo do container principal */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
/* Estilo do Card */
.card {
    background-color: #1a1a2e; /* Cor de fundo do card */
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
}
.card:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}
/* Esconde o rodapé 'Made with Streamlit' */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --- CONFIGURAÇÕES DO MODELO ---
MODEL_PATH = 'models/deepfake_detector_v3_finetuned.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- FUNÇÕES DO MODELO (com cache para performance) ---

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_resource
def load_face_detector():
    return mtcnn.MTCNN()

def predict_face(model, face_image):
    try:
        img = face_image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_ready = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)
        prediction = model.predict(img_ready, verbose=0)
        return prediction[0][0]
    except Exception as e:
        st.error(f"Erro ao realizar a previsão: {e}")
        return None

# Carrega os modelos
classifier_model = load_trained_model()
face_detector_model = load_face_detector()

# --- LAYOUT DA INTERFACE ---

# Barra Lateral
with st.sidebar:
    st.title("Sobre o Projeto")
    st.info("""
        Este é um detector de Deepfakes que utiliza um modelo de IA de ponta (EfficientNetB0) 
        com a técnica de Aprendizado por Transferência e Ajuste Fino para alcançar alta precisão.
    """)
    st.warning("O processamento de dados para o modelo V4 final ainda está em andamento.")

# Corpo Principal
st.title("🤖 Detector de Deepfakes v2.0")

# Card de Upload
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("1. Faça o Upload da Imagem")
    st.write("Envie uma imagem (JPG, JPEG, PNG) para que a IA possa analisá-la.")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)


if uploaded_file is not None and classifier_model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    detections = face_detector_model.detect_faces(opencv_image_rgb)

    # Card de Resultado
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("2. Resultado da Análise")

        if not detections:
            st.warning("Opa! Nenhum rosto foi detectado nesta imagem. Por favor, tente outra.")
        else:
            x, y, width, height = detections[0]['box']
            cropped_face = opencv_image_rgb[y:y+height, x:x+width]
            face_pil = Image.fromarray(cropped_face)

            col1, col2 = st.columns([1, 1.5]) # Colunas para imagem e resultado
            
            with col1:
                st.image(face_pil, caption='Rosto Detectado para Análise.', use_column_width=True)

            with col2:
                with st.spinner('Analisando o rosto...'):
                    prediction_score = predict_face(classifier_model, face_pil)
                
                if prediction_score is not None:
                    st.success("Análise concluída!")
                    score_percent = prediction_score * 100
                    st.metric(label="Pontuação de 'Realidade'", value=f"{score_percent:.2f}%")

                    if prediction_score > 0.5:
                        st.markdown(f"### Veredito: ✅ <span style='color:#32CD32;font-size:24px;'>PROVAVELMENTE REAL.</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### Veredito: ❌ <span style='color:#FF4B4B;font-size:24px;'>PROVAVELMENTE FAKE.</span>", unsafe_allow_html=True)
                    
                    st.info(f"**Como ler o resultado:** A 'Pontuação de Realidade' indica a confiança da IA de que a imagem é autêntica. Valores altos (próximos a 100%) sugerem um rosto real, enquanto valores baixos (próximos a 0%) sugerem um deepfake.", icon="ℹ️")

        st.markdown('</div>', unsafe_allow_html=True)