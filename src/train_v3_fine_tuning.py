import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint # <-- NOVO: Importamos o assistente
import os

# --- PARÂMETROS DE CONFIGURAÇÃO ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Caminhos dos dados e do modelo final
TRAIN_DIR = 'data/final_train'
VALIDATION_DIR = 'data/final_validation'
BASE_MODEL_PATH = 'models/deepfake_detector_v2_efficientnet.keras'
FINETUNED_MODEL_SAVE_PATH = 'models/deepfake_detector_v4_final.keras'

# Número de épocas para o ajuste fino
FINE_TUNE_EPOCHS = 10 
# ---------------------------------

# A preparação dos dados continua a mesma
train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest', preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
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

# Carregamento do modelo continua o mesmo
print(f"--- Carregando modelo base de: {BASE_MODEL_PATH} ---")
model = tf.keras.models.load_model(BASE_MODEL_PATH)
model.trainable = True
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Re-compilação continua a mesma
print("--- Re-compilando o modelo para ajuste fino com learning rate baixo ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- NOVO: CONFIGURAÇÃO DO CHECKPOINT ---
# Este assistente vai monitorar a acurácia de validação
# e salvar APENAS o melhor modelo encontrado durante o treinamento.
checkpoint_callback = ModelCheckpoint(
    filepath=FINETUNED_MODEL_SAVE_PATH, # Onde salvar o modelo
    monitor='val_accuracy',             # Métrica a ser observada
    verbose=1,                          # Mostra uma mensagem quando salva
    save_best_only=True,                # Salva apenas se for o melhor resultado até agora
    mode='max'                          # 'max' porque queremos maximizar a acurácia
)
# -----------------------------------------

# --- TREINAMENTO FINAL (COM O ASSISTENTE) ---
print("\n--- INICIANDO O TREINAMENTO FINAL COM CHECKPOINT ---")
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint_callback] # <-- Adicionamos nosso assistente aqui
)

# A linha model.save() não é mais estritamente necessária, pois o checkpoint já fez o trabalho
print("\n--- TREINAMENTO CONCLUÍDO. O MELHOR MODELO FOI SALVO AUTOMATICAMENTE. ---")