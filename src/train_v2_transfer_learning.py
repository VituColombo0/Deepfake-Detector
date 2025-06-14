import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
import os

# --- PARÂMETROS DE CONFIGURAÇÃO ---
IMG_HEIGHT = 224 # EfficientNet espera imagens maiores (224x224)
IMG_WIDTH = 224
BATCH_SIZE = 32

TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'
MODEL_SAVE_PATH = 'models/deepfake_detector_v2_efficientnet.keras'
# ---------------------------------

# 1. PREPARAÇÃO DOS DADOS (quase igual a antes)
# A única diferença é que cada modelo pré-treinado tem sua própria forma de 'normalizar'
# os pixels. O ImageDataGenerator lida com isso para nós.
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
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 2. CONSTRUÇÃO DO MODELO COM TRANSFER LEARNING
# Carrega o modelo EfficientNetB0 pré-treinado em milhões de imagens (dataset ImageNet)
# `include_top=False` significa que não queremos a camada final de classificação original.
# `weights='imagenet'` especifica que queremos os pesos (o conhecimento) pré-treinados.
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# "Congela" as camadas do modelo base. Não vamos retreiná-las.
# Elas já são especialistas em detectar formas, texturas, etc.
base_model.trainable = False

# Adiciona nossas próprias camadas no topo para a nossa tarefa específica
x = base_model.output
x = GlobalAveragePooling2D()(x) # Reduz a dimensionalidade
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
# A nossa camada final de decisão (1 neurônio, ativação sigmoid)
predictions = Dense(1, activation='sigmoid')(x)

# Este é o nosso novo modelo final
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# 3. COMPILAÇÃO DO MODELO
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. TREINAMENTO DO MODELO
print("\n--- INICIANDO O TREINAMENTO COM TRANSFER LEARNING ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10, # Geralmente, transfer learning requer menos épocas
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# 5. SALVANDO O MODELO V2
print("\n--- TREINAMENTO V2 CONCLUÍDO. SALVANDO O MODELO... ---")
model.save(MODEL_SAVE_PATH)
print(f"Modelo V2 salvo em: {MODEL_SAVE_PATH}")