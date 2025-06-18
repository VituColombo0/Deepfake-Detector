import cv2
import mtcnn
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os

# --- CONFIGURAÇÕES ---
# O modelo que vamos usar para a análise
MODEL_PATH = 'models/deepfake_detector_v3_finetuned.keras'
# O vídeo que vamos analisar
VIDEO_PATH = 'video_teste.mp4'  # Coloque um vídeo de teste na pasta principal do projeto
# Onde salvar o vídeo com o resultado
OUTPUT_VIDEO_PATH = 'resultado_video.mp4'

# Parâmetros de análise
IMG_HEIGHT = 224
IMG_WIDTH = 224
FRAME_INTERVAL = 5  # Analisar 1 a cada 5 frames para agilizar

# Cores para os retângulos (em formato BGR que o OpenCV usa)
COLOR_REAL = (0, 255, 0)  # Verde
COLOR_FAKE = (0, 0, 255)  # Vermelho
# --------------------

def main():
    """Função principal para processar o vídeo."""
    print("--- Carregando modelos... ---")
    # Carrega o detector de rostos e o classificador
    face_detector = mtcnn.MTCNN()
    classifier_model = tf.keras.models.load_model(MODEL_PATH)

    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {VIDEO_PATH}")
        return

    # Pega as propriedades do vídeo para criar o vídeo de saída
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define o codec e cria o objeto para escrever o vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    print(f"--- Processando vídeo: {VIDEO_PATH} ---")
    frame_idx = 0
    real_votes = 0
    fake_votes = 0
    
    # Usa o TQDM para criar uma barra de progresso para os frames do vídeo
    for _ in tqdm(range(total_frames), desc="Analisando Vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Apenas processa o frame no intervalo definido
        if frame_idx % FRAME_INTERVAL == 0:
            # Detecta rostos no frame
            detections = face_detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if detections:
                main_face = detections[0] # Pega o primeiro rosto
                x, y, w, h = main_face['box']
                
                # Recorta o rosto para análise
                cropped_face = frame[y:y+h, x:x+w]
                
                # Prepara a imagem do rosto para o classificador
                face_for_pred = cv2.resize(cropped_face, (IMG_WIDTH, IMG_HEIGHT))
                face_for_pred = image.img_to_array(face_for_pred)
                face_for_pred = np.expand_dims(face_for_pred, axis=0)
                face_for_pred = tf.keras.applications.efficientnet.preprocess_input(face_for_pred)

                # Faz a previsão
                prediction = classifier_model.predict(face_for_pred, verbose=0)[0][0]

                # Define o texto e a cor com base na previsão
                if prediction > 0.5:
                    label = f"REAL: {prediction:.2%}"
                    color = COLOR_REAL
                    real_votes += 1
                else:
                    label = f"FAKE: {1-prediction:.2%}"
                    color = COLOR_FAKE
                    fake_votes += 1
                
                # Desenha o retângulo e o texto no frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Escreve o frame (modificado ou não) no arquivo de saída
        out.write(frame)
        frame_idx += 1

    # Libera os recursos
    cap.release()
    out.release()
    
    print("\n--- Análise de vídeo concluída! ---")
    print(f"Vídeo com resultado salvo em: {OUTPUT_VIDEO_PATH}")
    total_votes = real_votes + fake_votes
    if total_votes > 0:
        real_percent = (real_votes / total_votes) * 100
        fake_percent = (fake_votes / total_votes) * 100
        print(f"Resultado geral: {real_percent:.2f}% dos frames analisados são REAL, {fake_percent:.2f}% são FAKE.")
        
        if fake_percent > 40: # Nosso critério (limiar)
             print("Veredito Final: O vídeo é provavelmente um DEEPFAKE.")
        else:
             print("Veredito Final: O vídeo é provavelmente REAL.")

if __name__ == '__main__':
    # Verifica se os arquivos necessários existem
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}. Treine o modelo primeiro.")
    elif not os.path.exists(VIDEO_PATH):
        print(f"Erro: Vídeo de teste não encontrado em {VIDEO_PATH}. Coloque um vídeo na pasta do projeto.")
    else:
        main()