
import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name='top_conv'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array_preprocessed)
        class_output = preds[0]
    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(original_image_cv, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap_resized = cv2.resize(heatmap, (original_image_cv.shape[1], original_image_cv.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    overlaid_image = cv2.addWeighted(original_image_cv, alpha, heatmap_colored, 1 - alpha, 0)
    return overlaid_image