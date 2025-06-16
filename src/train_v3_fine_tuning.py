import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- PARÂMETROS DE CONFIGURAÇÃO ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Caminhos dos dados e dos modelos
TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'
BASE_MODEL_PATH = 'models/deepfake_detector_v2_efficientnet.keras'
FINETUNED_MODEL_SAVE_PATH = 'models/deepfake_detector_v3_finetuned.keras'

# Número de épocas para o ajuste fino
FINE_TUNE_EPOCHS = 10
# ---------------------------------

# 1. PREPARAÇÃO DOS DADOS (exatamente como antes)
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


# 2. CARREGAR E PREPARAR O MODELO PARA AJUSTE FINO
print(f"--- Carregando modelo base de: {BASE_MODEL_PATH} ---")
model = tf.keras.models.load_model(BASE_MODEL_PATH)

# --- CORREÇÃO AQUI ---
# Em vez de tentar encontrar uma sub-camada, tornamos o modelo inteiro "treinável"
model.trainable = True

# É uma boa prática manter as camadas de BatchNormalization congeladas.
# Isso ajuda a estabilizar o treinamento e a manter o conhecimento do modelo base.
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
# ---------------------


# 3. RE-COMPILAR O MODELO COM UMA TAXA DE APRENDIZADO BAIXA
print("--- Re-compilando o modelo para ajuste fino com learning rate baixo ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Taxa de aprendizado bem baixa
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary() # O resumo agora mostrará muito mais parâmetros treináveis

# 4. CONTINUAR O TREINAMENTO (AJUSTE FINO)
print("\n--- INICIANDO O AJUSTE FINO (FINE-TUNING) ---")
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# 5. SALVANDO O MODELO V3 FINAL
print("\n--- AJUSTE FINO CONCLUÍDO. SALVANDO O MODELO V3... ---")
model.save(FINETUNED_MODEL_SAVE_PATH)
print(f"Modelo V3 (Fine-Tuned) salvo em: {FINETUNED_MODEL_SAVE_PATH}")