import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'models/deepfake_detector_model.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
# --------------------

def predict_image(image_path):
    """
    Carrega o modelo treinado, prepara uma imagem e prevê se é REAL ou FAKE.
    """
    print("--- Carregando o modelo treinado... ---")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar o modelo. Verifique o caminho: {MODEL_PATH}")
        print(f"Detalhe do erro: {e}")
        return

    print(f"--- Carregando e preparando a imagem: {image_path} ---")
    try:
        # Carrega a imagem e redimensiona para o tamanho que o modelo espera
        img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        
        # Converte a imagem para um array numpy
        img_array = image.img_to_array(img)
        
        # Normaliza os pixels da imagem (mesmo passo do treinamento)
        img_array /= 255.0
        
        # Adiciona uma dimensão extra, pois o modelo espera um 'lote' (batch) de imagens
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"[ERRO] Não foi possível carregar ou processar a imagem.")
        print(f"Detalhe do erro: {e}")
        return

    print("--- Realizando a previsão... ---")
    prediction = model.predict(img_array)

    # O gerador de dados atribuiu 'fake' a 0 e 'real' a 1.
    # A saída da rede 'sigmoid' é uma probabilidade perto de 0 ou 1.
    print("\n--- Resultado ---")
    print(f"Probabilidade Bruta: {prediction[0][0]:.6f}")

    if prediction[0][0] > 0.5:
        print("Veredito: Este rosto é provavelmente REAL.")
    else:
        print("Veredito: Este rosto é provavelmente FAKE.")

# --- PONTO DE ENTRADA DO SCRIPT ---
if __name__ == '__main__':
    # Pega o caminho da imagem passado como argumento no terminal
    if len(sys.argv) != 2:
        print("\nUso incorreto!")
        print("Como usar: python src/predict.py <caminho_para_sua_imagem>")
        sys.exit(1)
    
    image_to_predict = sys.argv[1]
    predict_image(image_to_predict)