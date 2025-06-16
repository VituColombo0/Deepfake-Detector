import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Deepfake",
    page_icon="🤖",
    layout="wide"
)

# --- CONFIGURAÇÕES DO MODELO ---
MODEL_PATH = 'models/deepfake_detector_v3_finetuned.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- FUNÇÕES DO MODELO ---

# Usamos @st.cache_resource para garantir que o modelo seja carregado apenas UMA VEZ.
@st.cache_resource
def load_trained_model():
    """Carrega o modelo treinado a partir do disco."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def predict_uploaded_image(model, uploaded_image):
    """Prepara a imagem e realiza a previsão."""
    try:
        # Prepara a imagem para o modelo
        img = Image.open(uploaded_image).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_ready = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)

        # Realiza a previsão
        prediction = model.predict(img_ready)
        
        return prediction[0][0]
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
        return None

# Carrega o modelo
model = load_trained_model()

# --- INTERFACE GRÁFICA ---
st.title("🤖 Detector de Deepfakes em Imagens")
st.write("""
    Faça o upload de uma imagem de rosto e a nossa Inteligência Artificial (Modelo V2 - EfficientNet)
    analisará se o rosto é REAL ou FAKE.
""")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Escolha uma imagem de rosto...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns([1, 2]) # Coluna da imagem menor que a da análise
    
    with col1:
        st.subheader("Imagem Enviada")
        # Corrigido o 'use_column_width' para 'use_container_width'
        st.image(uploaded_file, caption='Sua imagem.', use_column_width=True)

    with col2:
        st.subheader("Análise da IA")
        with st.spinner('Analisando a imagem...'):
            prediction_score = predict_uploaded_image(model, uploaded_file)
        
        if prediction_score is not None:
            st.success("Análise concluída!")
            
            # Formata o resultado de forma mais visual
            st.metric(label="Pontuação de 'Realidade'", value=f"{prediction_score:.2%}")

            if prediction_score > 0.5:
                st.markdown("### Veredito: <span style='color:green;font-size:24px;'>Este rosto é provavelmente REAL.</span>", unsafe_allow_html=True)
            else:
                st.markdown("### Veredito: <span style='color:red;font-size:24px;'>Este rosto é provavelmente FAKE.</span>", unsafe_allow_html=True)