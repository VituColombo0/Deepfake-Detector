

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import sys


TRAIN_DIR = 'data/final_train'
VALIDATION_DIR = 'data/final_validation'
BASE_MODEL_PATH = 'models/deepfake_detector_v6_ultimate.keras'
FINAL_MODEL_SAVE_PATH = 'models/deepfake_detector_v7_final.keras' 
IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS = 224, 224, 32, 30



if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VALIDATION_DIR):
    print(f"[ERRO] Diretórios de treino '{TRAIN_DIR}' ou validação '{VALIDATION_DIR}' não encontrados.")
    print("Execute o script 'prepare_final_dataset.py' primeiro.")
    sys.exit(1)

print(f"--- Carregando o modelo base ({os.path.basename(BASE_MODEL_PATH)}) ---")
try:
    model = tf.keras.models.load_model(BASE_MODEL_PATH)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    print("Modelo base carregado e recompilado para ajuste fino.")
except Exception as e:
    print(f"[ERRO FATAL] Não foi possível carregar o modelo base: {e}")
    sys.exit(1)


train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)
validation_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=True
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode='binary'
)


checkpoint = ModelCheckpoint(
    filepath=FINAL_MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max'
)
early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True
)
callbacks_list = [checkpoint, early_stopping]

print(f"\n--- Iniciando o treinamento do Modelo V7 por até {EPOCHS} épocas... ---")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks_list
)
print("\n--- Treinamento Concluído! O modelo do V7 foi salvo em", FINAL_MODEL_SAVE_PATH)