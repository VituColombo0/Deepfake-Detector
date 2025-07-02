# src/face_detector.py (Versão que aceita argumentos)

import os
import cv2
import sys
from tqdm import tqdm
import mtcnn
from multiprocessing import Pool, cpu_count

# Esta função continua a mesma
def init_worker():
    global face_detector
    face_detector = mtcnn.MTCNN()

# Esta função agora recebe a pasta de saída como argumento
def process_single_image(args):
    image_path, output_dir = args
    try:
        image = cv2.imread(image_path)
        if image is None: return None

        if not face_detector: init_worker() # Garante que o detector seja inicializado
            
        detections = face_detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if detections:
            main_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
            x, y, w, h = main_face['box']
            if w > 50 and h > 50:
                cropped_face = image[y:y+h, x:x+w]
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, cropped_face)
        return None
    except Exception as e:
        return f"Erro em {os.path.basename(image_path)}: {e}"

# A função principal agora é mais genérica
def process_image_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    files_to_process = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files_to_process:
        print(f"[AVISO] Nenhuma imagem encontrada em '{input_folder}'")
        return

    print(f"Encontradas {len(files_to_process)} imagens para processar em '{input_folder}'.")
    num_processes = cpu_count() - 1 if cpu_count() > 1 else 1
    print(f"Iniciando processamento com {num_processes} processos.")
    
    # Prepara os argumentos para cada chamada da função
    tasks = [(path, output_folder) for path in files_to_process]

    with Pool(processes=num_processes, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks)))
    
    errors = [r for r in results if r is not None]
    if errors:
        print("\n--- Ocorreram alguns erros durante o processamento: ---")
        for error_msg in errors:
            print(error_msg)
    print(f"--- Processamento de '{input_folder}' concluído! ---")

# O script agora lê os caminhos do terminal
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python face_detector.py <pasta_de_entrada> <pasta_de_saida>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    process_image_folder(input_path, output_path)