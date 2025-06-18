import cv2
import mtcnn
import os
import glob
from tqdm import tqdm # Importamos a biblioteca da barra de progresso

# A pasta de onde vamos ler as 200.000 fotos do CelebA
INPUT_DIR = 'data/img_align_celeba' 
# Uma nova pasta para salvar os rostos recortados do CelebA
OUTPUT_DIR = 'data/processed_celeba'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Encontra todos os arquivos de imagem no diretório de entrada
image_paths = glob.glob(os.path.join(INPUT_DIR, '*.jpg'))
print(f"Encontradas {len(image_paths)} imagens em '{INPUT_DIR}'")

face_detector = mtcnn.MTCNN()

# --- Loop com Barra de Progresso (tqdm) ---
# O tqdm vai envolver nossa lista de imagens e mostrar o progresso
for image_path in tqdm(image_paths, desc="Processando Imagens"):
    try:
        # --- Lógica para ser "Resumível" ---
        # Cria o nome do arquivo de saída esperado
        original_filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, original_filename)

        # Se o arquivo já existe na pasta de saída, pula para o próximo
        if os.path.exists(output_path):
            continue
        # -----------------------------------------

        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = face_detector.detect_faces(image_rgb)
        
        if detections:
            # Pega apenas o primeiro rosto e de maior confiança
            main_face = detections[0]
            x, y, width, height = main_face['box']
            
            # Recorta o rosto da imagem original
            cropped_face = image[y:y+height, x:x+width]

            # Salva o rosto recortado
            cv2.imwrite(output_path, cropped_face)

    except Exception as e:
        # Se ocorrer um erro em uma imagem específica (ex: arquivo corrompido),
        # ele registrará o erro e continuará para a próxima, sem travar.
        print(f"Erro ao processar {image_path}: {e}")
        continue

print("\n--- Processamento em lote concluído! ---")