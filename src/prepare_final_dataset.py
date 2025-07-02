# src/prepare_final_dataset.py

import os
import glob
import random
import shutil
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
# Lista de TODAS as pastas de origem, incluindo o novo dataset.
SOURCE_FOLDERS = {
    'real': [
        'data/processed',             # Rostos do UTKFace
        'data/processed_celeba',      # Rostos do CelebA
        'data/processed_celecdf/real', # Rostos REAIS do Celeb-DF
        'data/processed_novo/train/real',
        'data/processed_novo/validation/real',
        'data/processed_novo/test/real'
    ],
    'fake': [
        'data/train/fake',              # Fakes antigos (StyleGAN)
        'data/processed_celecdf/fake',  # Fakes do Celeb-DF
        'data/processed_dftimit/fake',  # Fakes do Deepfake-TIMIT
        'data/processed_novo/train/fake',
        'data/processed_novo/validation/fake',
        'data/processed_novo/test/fake'
    ]
}

TRAIN_FOLDER = 'data/final_train'
VALIDATION_FOLDER = 'data/final_validation'
SPLIT_RATIO = 0.9
# --------------------

def collect_files(source_list):
    all_files = []
    print(f"Coletando arquivos de: {source_list}")
    for folder in source_list:
        if not os.path.isdir(folder):
            print(f"  [AVISO] Pasta não encontrada: {folder}. Pulando.")
            continue
        # Busca recursivamente por imagens para garantir que nada seja perdido
        files = glob.glob(os.path.join(folder, '**', '*.jp*g'), recursive=True)
        files.extend(glob.glob(os.path.join(folder, '**', '*.png'), recursive=True))
        all_files.extend(files)
    return all_files

def balance_and_split_data(real_files, fake_files):
    if not real_files or not fake_files:
        print("[ERRO] Uma das classes (real ou fake) não tem imagens. Abortando.")
        return None
    
    min_size = min(len(real_files), len(fake_files))
    print(f"\nBalanceando datasets. Usando {min_size} amostras de cada classe.")
    
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    real_files_sample = real_files[:min_size]
    fake_files_sample = fake_files[:min_size]

    datasets = {'real': {}, 'fake': {}}
    real_split_point = int(len(real_files_sample) * SPLIT_RATIO)
    datasets['real']['train'] = real_files_sample[:real_split_point]
    datasets['real']['validation'] = real_files_sample[real_split_point:]
    
    fake_split_point = int(len(fake_files_sample) * SPLIT_RATIO)
    datasets['fake']['train'] = fake_files_sample[:fake_split_point]
    datasets['fake']['validation'] = fake_files_sample[fake_split_point:]
    
    return datasets

def copy_files(datasets):
    if datasets is None: return
    for label, splits in datasets.items():
        for split_name, files in splits.items():
            dest_folder = os.path.join(TRAIN_FOLDER if split_name == 'train' else VALIDATION_FOLDER, label)
            os.makedirs(dest_folder, exist_ok=True)
            print(f"\nCopiando {len(files)} arquivos para '{dest_folder}'...")
            for f in tqdm(files, desc=f"Copiando {label} {split_name}"):
                shutil.copy(f, dest_folder)

if __name__ == '__main__':
    print("--- INICIANDO A PREPARAÇÃO DO SUPER-DATASET FINAL ---")
    if os.path.exists(TRAIN_FOLDER): shutil.rmtree(TRAIN_FOLDER)
    if os.path.exists(VALIDATION_FOLDER): shutil.rmtree(VALIDATION_FOLDER)
    print("Pastas de destino antigas foram limpas.")

    real_files = collect_files(SOURCE_FOLDERS['real'])
    fake_files = collect_files(SOURCE_FOLDERS['fake'])
    
    balanced_datasets = balance_and_split_data(real_files, fake_files)
    copy_files(balanced_datasets)
    
    print("\n--- PREPARAÇÃO DO SUPER-DATASET CONCLUÍDA ---")