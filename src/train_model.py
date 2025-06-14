import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# --- PARÂMETROS DE CONFIGURAÇÃO ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32  # Quantas imagens o modelo vê de uma vez

TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'
MODEL_SAVE_PATH = 'models/deepfake_detector_model.keras'
# ---------------------------------

# 1. PREPARAÇÃO DOS DADOS
# Cria 'geradores' que vão ler as imagens das pastas e prepará-las para o modelo.
# O 'rescale' normaliza os pixels das imagens (de 0-255 para 0-1).
# A 'data augmentation' no treino (rotation, zoom) cria variações das imagens 
# para o modelo generalizar melhor e não 'decorar'.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary' # 'binary' porque temos 2 classes: real e fake
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 2. CONSTRUÇÃO DO MODELO (ARQUITETURA DA CNN)
model = Sequential([
    # 1ª Camada de Convolução: 32 filtros para encontrar padrões básicos
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    # 2ª Camada de Convolução: 64 filtros para encontrar padrões mais complexos
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # 3ª Camada de Convolução
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Achata os dados para prepará-los para as camadas de decisão
    Flatten(),

    # Camada de 'cérebro' com 512 neurônios
    Dense(512, activation='relu'),
    
    # Camada de Dropout para evitar 'overfitting' (o modelo decorar em vez de aprender)
    Dropout(0.5),

    # Camada final de decisão: 1 neurônio que dará a probabilidade de ser 'fake' (0) ou 'real' (1)
    Dense(1, activation='sigmoid')
])

# Mostra um resumo da arquitetura que criamos
model.summary()

# 3. COMPILAÇÃO DO MODELO
# Define as regras do aprendizado: otimizador, função de perda e métrica de sucesso.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. TREINAMENTO DO MODELO
print("\n--- INICIANDO O TREINAMENTO ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=15, # Quantas vezes o modelo verá o dataset de treino inteiro
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# 5. SALVANDO O MODELO TREINADO
print("\n--- TREINAMENTO CONCLUÍDO. SALVANDO O MODELO... ---")
# Garante que a pasta 'models' exista
os.makedirs('models', exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Modelo salvo em: {MODEL_SAVE_PATH}")