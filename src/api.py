# src/api.py

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import mtcnn
import cv2
import os
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'models/deepfake_detector_v4_final.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224

# --- CARREGAMENTO DOS MODELOS ---
print("--- Carregando modelos de IA... ---")
face_detector = mtcnn.MTCNN()
classifier_model = tf.keras.models.load_model(MODEL_PATH)
print("--- Modelos carregados com sucesso! ---")

# --- FUNÇÕES AUXILIARES ---
def predict_face(model, face_pil):
    img = face_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_ready = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)
    prediction = model.predict(img_ready, verbose=0)
    return prediction[0][0]

# --- ROTAS DA API ---

@app.route('/predict', methods=['POST'])
def handle_image_prediction():
    if 'file' not in request.files: return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    try:
        image_pil = Image.open(file.stream).convert('RGB')
        image_cv = np.array(image_pil)
        detections = face_detector.detect_faces(image_cv)
        if not detections: return jsonify({'error': 'Nenhum rosto detectado'}), 400
        
        main_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        x, y, w, h = main_face['box']
        face_pil = Image.fromarray(image_cv[y:y+h, x:x+w])
        
        score = predict_face(classifier_model, face_pil)
        verdict = "REAL" if score > 0.5 else "FAKE"
        confidence = score if verdict == "REAL" else 1 - score
        
        return jsonify({'verdict': verdict, 'confidence': f"{confidence:.2%}"})
    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro: {str(e)}'}), 500

@app.route('/predict_video', methods=['POST'])
def handle_video_prediction():
    if 'file' not in request.files: return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    # Salva o vídeo temporariamente
    temp_video_path = f"temp_{int(time.time())}_{file.filename}"
    file.save(temp_video_path)
    
    try:
        cap = cv2.VideoCapture(temp_video_path)
        real_votes, fake_votes, frame_count = 0, 0, 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % 5 == 0: # Analisa a cada 5 frames
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = face_detector.detect_faces(frame_rgb)
                if detections:
                    main_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                    x, y, w, h = main_face['box']
                    face_pil = Image.fromarray(frame_rgb[y:y+h, x:x+w])
                    score = predict_face(classifier_model, face_pil)
                    if score > 0.5: real_votes += 1
                    else: fake_votes += 1
            frame_count += 1
        
        cap.release()
        os.remove(temp_video_path) # Limpa o arquivo temporário

        total_votes = real_votes + fake_votes
        if total_votes == 0: return jsonify({'error': 'Nenhum rosto detectado no vídeo'}), 400

        fake_percent = (fake_votes / total_votes)
        verdict = "FAKE" if fake_percent > 0.4 else "REAL" # Nosso limiar
        confidence = fake_percent if verdict == "FAKE" else 1 - fake_percent
        
        return jsonify({'verdict': verdict, 'confidence': f"{confidence:.2%}"})
        
    except Exception as e:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        return jsonify({'error': f'Ocorreu um erro: {str(e)}'}), 500

# --- PONTO DE ENTRADA DO SERVIDOR ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)