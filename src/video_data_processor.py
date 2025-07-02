# src/video_data_processor.py (Versão Recursiva)

import cv2
import os
import mtcnn
import sys
import glob # Usaremos o glob para uma busca mais poderosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURAÇÕES ---
# Configure para a pasta que você quer processar
# Alvo: Deepfake-TIMIT de Baixa Qualidade
INPUT_VIDEO_FOLDER = 'data/deepfake_TIMIT/lower_quality' 

# A pasta de saída continua a mesma
OUTPUT_FACES_FOLDER = 'data/processed_dftimit/fake'
MAX_FACES_PER_VIDEO = 5 
# --------------------

face_detector = None

def init_worker():
    """Inicializa o detector de rostos em cada processo."""
    global face_detector
    # print("Inicializando detector de rostos...") # Desativado para um log mais limpo
    face_detector = mtcnn.MTCNN()

def process_single_video(video_path):
    """Processa um único arquivo de vídeo."""
    try:
        video_name = os.path.basename(video_path)
        if not face_detector:
            return f"Detector não inicializado para {video_name}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return f"Não abriu {video_name}"

        frame_idx = 0
        faces_saved = 0
        while cap.isOpened() and faces_saved < MAX_FACES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret: break
            
            detections = face_detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if detections:
                main_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                x, y, w, h = main_face['box']
                if w > 50 and h > 50:
                    cropped_face = frame[y:y+h, x:x+w]
                    video_name_no_ext = os.path.splitext(video_name)[0]
                    face_filename = f"{video_name_no_ext}_frame{frame_idx}.jpg"
                    save_path = os.path.join(OUTPUT_FACES_FOLDER, face_filename)
                    cv2.imwrite(save_path, cropped_face)
                    faces_saved += 1
            frame_idx += 1
        
        cap.release()
        return None
    except Exception as e:
        return f"Erro ao processar {os.path.basename(video_path)}: {e}"

if __name__ == '__main__':
    print("--- INICIANDO PIPELINE DE PROCESSAMENTO RECURSIVO ---")
    
    # Garante que a pasta de saída exista
    os.makedirs(OUTPUT_FACES_FOLDER, exist_ok=True)
    
    # --- MUDANÇA PRINCIPAL AQUI ---
    # Usa o glob para encontrar TODOS os vídeos em TODAS as subpastas
    print(f"Buscando vídeos recursivamente em: '{INPUT_VIDEO_FOLDER}'")
    extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_FOLDER, '**', ext), recursive=True))
    # -----------------------------

    if not video_files:
        print(f"Nenhum vídeo encontrado.")
    else:
        print(f"Encontrados {len(video_files)} vídeos para processar.")
        num_processes = cpu_count() - 1 if cpu_count() > 1 else 1
        print(f"Iniciando processamento paralelo com {num_processes} processos.")

        with Pool(processes=num_processes, initializer=init_worker) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_video, video_files), total=len(video_files)))
        
        errors = [r for r in results if r is not None]
        if errors:
            print("\n--- Ocorreram alguns erros durante o processamento: ---")
            for error_msg in errors:
                print(error_msg)

    print("\n--- PROCESSAMENTO DE VÍDEOS CONCLUÍDO ---")