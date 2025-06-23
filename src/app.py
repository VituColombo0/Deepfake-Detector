import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import mtcnn
import time
import requests
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
# Garante que o nosso script de XAI seja encontrado
from xai_utils import generate_gradcam_heatmap, overlay_heatmap_on_image 

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Deepfake Detector Pro",
    page_icon="🤖",
    layout="wide"
)

# --- TEMA E ESTILO ---
st.markdown("""
<style>
    .block-container { padding: 2rem 3rem; }
    .card { background-color: #0E1117; border: 1px solid #262730; border-radius: 15px; padding: 25px; margin-top: 20px; box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2); }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÕES DO MODELO ---
# ATENÇÃO: Quando o V4 estiver pronto, trocaremos para 'deepfake_detector_v4_final.keras'
MODEL_PATH = 'models/deepfake_detector_v3_finetuned.keras' 
IMG_HEIGHT, IMG_WIDTH = 224, 224

# --- FUNÇÕES CACHEADAS ---
@st.cache_resource
def load_models():
    face_detector = mtcnn.MTCNN()
    classifier_model = None
    if os.path.exists(MODEL_PATH):
        try:
            classifier_model = tf.keras.models.load_model(MODEL_PATH)
            classifier_model.compile(run_eagerly=True)
        except Exception as e:
            st.error(f"Erro ao carregar o modelo classificador: {e}")
    return face_detector, classifier_model

@st.cache_data
def load_lottie_animation(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

def preprocess_image_for_model(pil_image):
    img = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)

# --- CARREGAMENTO INICIAL ---
face_detector, classifier_model = load_models()
lottie_animation = load_lottie_animation("https://assets9.lottiefiles.com/packages/lf20_x2oi05jr.json")

# --- INTERFACE PRINCIPAL ---
st.title("👁️ Deepfake Detector Pro")
st.write("Uma ferramenta de IA para analisar e detectar manipulações em imagens e vídeos.")
st.markdown("---")

if classifier_model is None:
    st.error("Modelo de IA não pôde ser carregado. Verifique se o modelo foi treinado e o caminho está correto.")
    st.stop()

tab1, tab2 = st.tabs(["🖼️ Análise de Imagem com XAI", "🎬 Análise de Vídeo com XAI"])

# --- LÓGICA COMPLETA DA ABA DE IMAGEM ---
with tab1:
    with st.container(border=True):
        st.header("1. Faça o Upload da Imagem")
        uploaded_image_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_image_file:
        with st.container(border=True):
            st.header("2. Resultado da Análise Explicável (XAI)")
            
            original_pil = Image.open(uploaded_image_file).convert('RGB')
            original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)

            with st.spinner("Detectando rosto..."):
                detections = face_detector.detect_faces(np.array(original_pil))

            if not detections:
                st.warning("Opa! Nenhum rosto foi detectado nesta imagem.")
            else:
                main_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                x, y, w, h = main_face['box']
                cropped_pil = original_pil.crop((x, y, x + w, y + h))
                
                with st.spinner("Analisando o rosto e gerando explicação..."):
                    img_ready_for_model = preprocess_image_for_model(cropped_pil)
                    prediction = classifier_model.predict(img_ready_for_model, verbose=0)[0][0]
                    heatmap = generate_gradcam_heatmap(img_ready_for_model, classifier_model)
                    
                    cropped_cv_for_overlay = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                    overlaid_image = overlay_heatmap_on_image(cropped_cv_for_overlay, heatmap)

                st.success("Análise concluída!")
                st.markdown("---")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(cropped_pil, caption='Rosto Detectado', use_column_width=True)
                with col2:
                    st.image(heatmap, caption='Mapa de Calor (Foco da IA)', use_column_width=True, channels="BGR")
                with col3:
                    st.image(overlaid_image, caption='Análise Sobreposta', use_column_width=True, channels="BGR")
                
                st.markdown("---")
                if prediction > 0.5:
                    st.success(f"Veredito: REAL (Confiança de {prediction:.2%})", icon="✅")
                    st.info("O mapa de calor mostra as áreas que a IA identificou como 'normais' ou 'autênticas'.", icon="💡")
                else:
                    st.error(f"Veredito: FAKE (Confiança de {1-prediction:.2%})", icon="❌")
                    st.info("O mapa de calor mostra as áreas que a IA considerou 'suspeitas' ou 'inconsistentes'.", icon="💡")

# --- LÓGICA COMPLETA DA ABA DE VÍDEO ---
with tab2:
    with st.container(border=True):
        st.header("Analisar um Arquivo de Vídeo")
        uploaded_video_file = st.file_uploader("Escolha um arquivo de vídeo...", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")

    if uploaded_video_file:
        if st.button("Iniciar Análise do Vídeo", key="analyze_video_button"):
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_video_path = os.path.join(temp_dir, f"{int(time.time())}_{uploaded_video_file.name}")
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video_file.getbuffer())

            progress_bar = st.progress(0, text="Análise em andamento... 0%")
            status_text = st.empty()
            image_placeholder = st.empty()

            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx, real_votes, fake_votes = 0, 0, 0
            suspicious_frames = [] 

            status_text.info("Processando vídeo... por favor, aguarde.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if frame_idx % 5 == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = face_detector.detect_faces(frame_rgb)
                    
                    if detections:
                        main_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                        x, y, w, h = main_face['box']
                        cropped_face = frame_rgb[y:y+h, x:x+w]
                        face_pil = Image.fromarray(cropped_face)
                        
                        img_ready = preprocess_image_for_model(face_pil)
                        prediction = classifier_model.predict(img_ready, verbose=0)[0][0]
                        
                        if prediction is not None:
                            if prediction > 0.5:
                                label, color, real_votes = f"REAL {prediction:.1%}", (0, 255, 0), real_votes + 1
                            else:
                                label, color, fake_votes = f"FAKE {1-prediction:.1%}", (0, 0, 255), fake_votes + 1
                                if (1 - prediction) > 0.6: # Guarda se a confiança de FAKE for alta
                                    suspicious_frames.append({'frame': frame_rgb, 'score': 1 - prediction})
                            
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_idx += 1
                progress_percent = int((frame_idx / total_frames) * 100)
                progress_bar.progress(progress_percent, text=f"Análise em andamento... {progress_percent}%")
            
            cap.release()
            os.remove(temp_video_path)
            progress_bar.empty()
            status_text.success("Análise de vídeo concluída!")
            
            with st.container(border=True):
                st.subheader("Relatório Final do Vídeo")
                # ... (resto da lógica de relatório e XAI do vídeo)