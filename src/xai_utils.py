# src/xai_utils.py

import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
    """
    Gera um mapa de calor Grad-CAM para uma dada imagem e modelo.
    """
    # Cria um sub-modelo que retorna tanto a saída da última camada convolucional
    # quanto a previsão final do modelo principal.
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Usa o GradientTape para calcular os gradientes
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # Pega a previsão da classe que queremos explicar (no nosso caso, a única saída)
        class_output = preds[0]

    # Calcula o gradiente da nossa previsão em relação à saída da última camada conv.
    grads = tape.gradient(class_output, last_conv_layer_output)

    # "Pesa" os canais do mapa de ativação pela importância (gradiente)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normaliza o mapa de calor para ficar entre 0 e 1 e aplica ReLU
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(original_image_cv, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Aplica o mapa de calor sobre a imagem original.
    """
    # Redimensiona o mapa de calor para o tamanho da imagem original
    heatmap_resized = cv2.resize(heatmap, (original_image_cv.shape[1], original_image_cv.shape[0]))
    
    # Converte para 8-bit e aplica o colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Mistura a imagem original com o mapa de calor
    overlaid_image = cv2.addWeighted(original_image_cv, alpha, heatmap_colored, 1 - alpha, 0)
    
    return overlaid_image