import cv2
import mtcnn
import os
import glob

INPUT_DIR = 'data/raw_images'
OUTPUT_DIR = 'data/processed'

# Garante que o diretório de saída exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = glob.glob(os.path.join(INPUT_DIR, '*.[jp][pn]g'))
print(f"--- Encontradas {len(image_paths)} imagens para processar ---")

# Inicializa o detector de rostos fora do loop por eficiência
face_detector = mtcnn.MTCNN()

for path in image_paths:
    print(f"\nProcessando imagem: {path}")
    
    image = cv2.imread(path)
    if image is None:
        print(f"  [ERRO] Não foi possível ler a imagem. Pulando.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = face_detector.detect_faces(image_rgb)
    
    if not detections:
        print("  Nenhum rosto detectado.")
        continue

    print(f"  Encontrado(s) {len(detections)} rosto(s).")

    # Cria um nome de arquivo de saída baseado no nome original
    base_filename = os.path.basename(path)
    filename_no_ext = os.path.splitext(base_filename)[0]

    for i, face in enumerate(detections):
        x, y, width, height = face['box']
        cropped_face = image[y:y+height, x:x+width]

        output_filename = os.path.join(OUTPUT_DIR, f'{filename_no_ext}_face_{i+1}.jpg')
        cv2.imwrite(output_filename, cropped_face)
        print(f"  Rosto salvo em: {output_filename}")

print("\n--- Processamento concluído! ---")