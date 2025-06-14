import os
import glob
import random
import shutil

# --- CONFIGURAÇÕES ---
# Mude estes caminhos se for usar para os rostos FAKES depois
SOURCE_FOLDER = 'data/temp_fake_faces' # <-- Onde estão suas imagens FAKE baixadas
TRAIN_FOLDER = 'data/train/fake'
VALIDATION_FOLDER = 'data/validation/fake'

# Proporção da divisão (0.8 = 80% para treino, 20% para validação)
SPLIT_RATIO = 0.8
# --------------------


def distribute_files():
    """
    Encontra, embaralha e distribui arquivos de uma pasta de origem
    para as pastas de treino e validação.
    """
    print(f"Iniciando organização da pasta: {SOURCE_FOLDER}")

    # Garante que as pastas de destino existam
    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(VALIDATION_FOLDER, exist_ok=True)

    # Encontra todos os arquivos de imagem na pasta de origem
    all_files = glob.glob(os.path.join(SOURCE_FOLDER, '*.[jp][pn]g'))
    
    if not all_files:
        print(f"[AVISO] Nenhuma imagem encontrada em {SOURCE_FOLDER}. O script não fará nada.")
        return

    # Embaralha a lista de arquivos de forma aleatória
    random.shuffle(all_files)
    print(f"Total de {len(all_files)} arquivos encontrados e embaralhados.")

    # Calcula o ponto de divisão
    split_point = int(len(all_files) * SPLIT_RATIO)

    # Divide a lista em treino e validação
    train_files = all_files[:split_point]
    validation_files = all_files[split_point:]

    # Move os arquivos para as pastas de destino
    print(f"\nMovendo {len(train_files)} arquivos para {TRAIN_FOLDER}...")
    for file_path in train_files:
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(TRAIN_FOLDER, file_name))

    print(f"\nMovendo {len(validation_files)} arquivos para {VALIDATION_FOLDER}...")
    for file_path in validation_files:
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(VALIDATION_FOLDER, file_name))

    print("\nOrganização concluída com sucesso!")


# Roda a função principal
if __name__ == "__main__":
    distribute_files()