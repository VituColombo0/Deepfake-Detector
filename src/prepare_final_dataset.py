import os
import glob
import random
import shutil
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
# Pastas contendo os rostos REAIS processados
REAL_FACE_SOURCES = [
    'data/processed',       # Rostos do UTKFace
    'data/processed_celeba' # Rostos do CelebA
]

# Pasta de onde vamos pegar os rostos FAKES.
# Vamos pegar da nossa pasta de treino antiga, que já está organizada.
FAKE_FACE_SOURCE = 'data/train/fake' 

# Pastas de destino final
FINAL_TRAIN_DIR = 'data/final_train'
FINAL_VALIDATION_DIR = 'data/final_validation'

# Proporção da divisão
SPLIT_RATIO = 0.8
# --------------------

def create_dirs():
    """Cria as pastas de destino final."""
    for folder in [FINAL_TRAIN_DIR, FINAL_VALIDATION_DIR]:
        os.makedirs(os.path.join(folder, 'real'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'fake'), exist_ok=True)
    print("Pastas de destino final criadas.")

# A função agora recebe as pastas de destino como argumentos
def organize_class(source_folders, class_name, train_dest_folder, validation_dest_folder):
    """Junta, embaralha e distribui os arquivos de uma classe (real ou fake)."""
    print(f"\nOrganizando a classe: '{class_name}'")
    
    all_files = []
    for source_folder in source_folders:
        print(f"Coletando arquivos de: {source_folder}")
        # Ajuste para pegar jpg e jpeg
        files = glob.glob(os.path.join(source_folder, '*.jp*g'))
        all_files.extend(files)
    
    if not all_files:
        print(f"[AVISO] Nenhum arquivo encontrado para a classe '{class_name}'. Pulando.")
        return

    random.shuffle(all_files)
    print(f"Total de {len(all_files)} arquivos encontrados e embaralhados.")

    split_point = int(len(all_files) * SPLIT_RATIO)
    train_files = all_files[:split_point]
    validation_files = all_files[split_point:]

    # Copia os arquivos de treino para o destino correto
    print(f"Copiando {len(train_files)} arquivos de treino...")
    for file_path in tqdm(train_files, desc=f"Train {class_name}"):
        file_name = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(train_dest_folder, class_name, file_name))

    # Copia os arquivos de validação para o destino correto
    print(f"Copiando {len(validation_files)} arquivos de validação...")
    for file_path in tqdm(validation_files, desc=f"Validation {class_name}"):
        file_name = os.path.basename(file_path)
        # --- AQUI ESTAVA O ERRO, AGORA CORRIGIDO ---
        shutil.copy(file_path, os.path.join(validation_dest_folder, class_name, file_name))
    
    print(f"Classe '{class_name}' organizada com sucesso.")


if __name__ == "__main__":
    create_dirs()
    # Chamamos a função passando as pastas de destino corretas
    organize_class(REAL_FACE_SOURCES, 'real', FINAL_TRAIN_DIR, FINAL_VALIDATION_DIR)
    organize_class([FAKE_FACE_SOURCE], 'fake', FINAL_TRAIN_DIR, FINAL_VALIDATION_DIR)
    print("\n--- Preparação do dataset final concluída! ---")