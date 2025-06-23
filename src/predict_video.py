import cv2
import mtcnn
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os

# --- CONFIGURAÇÕES ---
# Usaremos o V3 por enquanto. Quando o V4 estiver pronto, só trocaremos este caminho.
MODEL_PATH = 'models/deepfake_detector_v3_finetuned.keras'

# Coloque um vídeo de teste na pasta principal do projeto e atualize o nome aqui.
VIDEO_PATH = 'video_teste.mp4'  
OUTPUT_VIDEO_PATH = 'resultado_video.mp4'

# Parâmetros de análise
IMG_HEIGHT = 224
IMG_WIDTH = 224
FRAME_INTERVAL = 3  # Analisar 1 a cada 3 frames para um bom equilíbrio entre velocidade e precisão.

# Cores para os retângulos (em formato BGR que o OpenCV usa)
COLOR_REAL = (0, 255, 0)  # Verde
COLOR_FAKE = (0, 0, 255)  # Vermelho
# --------------------

def main():
    """Função principal para processar o vídeo."""
    print("--- Carregando modelos... ---")
    try:
        face_detector = mtcnn.MTCNN()
        classifier_model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar os modelos: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    print(f"--- Processando vídeo: {VIDEO_PATH} ---")
    frame_idx = 0
    real_votes = 0
    fake_votes = 0
    
    with tqdm(total=total_frames, desc="Analisando Vídeo") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apenas processa o frame no intervalo definido
            if frame_idx % FRAME_INTERVAL == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = face_detector.detect_faces(frame_rgb)
                
                if detections:
                    # Pega o rosto com maior área na tela
                    main_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                    x, y, w, h = main_face['box']
                    
                    cropped_face = frame_rgb[y:y+h, x:x+w]
                    
                    face_for_pred = cv2.resize(cropped_face, (IMG_WIDTH, IMG_HEIGHT))
                    face_for_pred = image.img_to_array(face_for_pred)
                    face_for_pred = np.expand_dims(face_for_pred, axis=0)
                    face_for_pred = tf.keras.applications.efficientnet.preprocess_input(face_for_pred)

                    prediction = classifier_model.predict(face_for_pred, verbose=0)[0][0]

                    if prediction > 0.5:
                        label = f"REAL: {prediction:.1%}"
                        color = COLOR_REAL
                        real_votes += 1
                    else:
                        label = f"FAKE: {1-prediction:.1%}"
                        color = COLOR_FAKE
                        fake_votes += 1
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    
    print("\n--- Análise de vídeo concluída! ---")
    print(f"Vídeo com resultado salvo em: {OUTPUT_VIDEO_PATH}")
    total_votes = real_votes + fake_votes
    if total_votes > 0:
        real_percent = (real_votes / total_votes) * 100
        fake_percent = (fake_votes / total_votes) * 100
        print(f"Resultado geral: {real_percent:.2f}% dos frames analisados são REAL, {fake_percent:.2f}% são FAKE.")
        
        if fake_percent > 40: # Nosso critério (limiar) para considerar o vídeo como um todo FAKE
             print("Veredito Final: O vídeo é provavelmente um DEEPFAKE.")
        else:
             print("Veredito Final: O vídeo é provavelmente REAL.")

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}. Treine o modelo primeiro.")
    elif not os.path.exists(VIDEO_PATH):
        print(f"Erro: Vídeo de teste não encontrado em {VIDEO_PATH}. Coloque um vídeo na pasta do projeto e ajuste o caminho no script.")
    else:
        main()