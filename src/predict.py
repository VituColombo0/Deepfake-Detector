import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os
import glob # <-- Importamos a biblioteca para encontrar arquivos

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'models/deepfake_detector_v2_efficientnet.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224
# --------------------

def predict_image(image_path):
    """
    Carrega o modelo treinado, prepara uma imagem e prevê se é REAL ou FAKE.
    """
    if not os.path.exists(image_path):
        print(f"[ERRO] O arquivo de imagem não foi encontrado em: {image_path}")
        return
        
    print("--- Carregando o modelo treinado... ---")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"--- Carregando e preparando a imagem: {image_path} ---")
    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_ready = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)

    print("--- Realizando a previsão... ---")
    prediction = model.predict(img_ready)

    print("\n--- Resultado ---")
    print(f"Probabilidade Bruta: {prediction[0][0]:.6f}")

    if prediction[0][0] > 0.5:
        print("Veredito: Este rosto é provavelmente REAL.")
    else:
        print("Veredito: Este rosto é provavelmente FAKE.")

# --- PONTO DE ENTRADA DO SCRIPT ---
if __name__ == '__main__':
    # Se um caminho de imagem foi fornecido no terminal, use-o.
    if len(sys.argv) == 2:
        image_to_predict = sys.argv[1]
        predict_image(image_to_predict)
    # Se NENHUM caminho foi fornecido, roda o teste automático.
    else:
        print("--- Nenhum arquivo especificado. Rodando teste de sanidade automático... ---")
        validation_real_path = 'data/validation/real/'
        
        # Encontra todos os arquivos de imagem na pasta
        test_files = glob.glob(os.path.join(validation_real_path, '*.[jp][pn]g'))
        
        # Se encontrou algum arquivo, pega o primeiro da lista para testar
        if test_files:
            first_file = test_files[0]
            predict_image(first_file)
        else:
            print(f"[ERRO] Nenhum arquivo de imagem encontrado em '{validation_real_path}' para o teste automático.")