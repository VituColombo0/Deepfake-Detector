# src/process_all_datasets.py

import os
import subprocess
import sys

def run_job(job_name, input_folder, output_folder):
    """Executa um único trabalho de processamento de imagens."""
    
    # Garante que a pasta de saída exista
    os.makedirs(output_folder, exist_ok=True)
    
    # Comando para chamar o nosso extrator de rostos de imagens
    command = [sys.executable, "src/face_detector.py", input_folder, output_folder]
    
    print(f"\n{'='*20}\n--- INICIANDO TAREFA: {job_name} ---\n{'='*20}")
    
    try:
        # Roda o script e espera ele terminar
        subprocess.run(command, check=True, text=True)
        print(f"\n--- TAREFA CONCLUÍDA: {job_name} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO FATAL] A tarefa '{job_name}' falhou: {e}")
        return False
    except FileNotFoundError:
        print(f"\n[ERRO FATAL] Não foi possível encontrar 'src/face_detector.py'.")
        return False

if __name__ == '__main__':
    # --- LISTA DE TAREFAS DE PROCESSAMENTO ---
    # IMPORTANTE: Confirme se este caminho para a sua pasta "Dataset" está correto.
    BASE_INPUT_PATH = 'C:/Users/LAOB/Downloads/Dataset' # <<< CONFIRME ESTA LINHA

    # Verifica se o caminho base existe antes de definir os trabalhos
    if not os.path.isdir(BASE_INPUT_PATH):
        print(f"[ERRO] O caminho base '{BASE_INPUT_PATH}' não foi encontrado.")
        print("Por favor, edite o script 'process_all_datasets.py' com o caminho correto.")
        sys.exit(1)

    JOBS = [
        {'name': 'Train Fake', 'input': os.path.join(BASE_INPUT_PATH, 'Train', 'Fake'), 'output': 'data/processed_novo/train/fake'},
        {'name': 'Train Real', 'input': os.path.join(BASE_INPUT_PATH, 'Train', 'Real'), 'output': 'data/processed_novo/train/real'},
        {'name': 'Validation Fake', 'input': os.path.join(BASE_INPUT_PATH, 'Validation', 'Fake'), 'output': 'data/processed_novo/validation/fake'},
        {'name': 'Validation Real', 'input': os.path.join(BASE_INPUT_PATH, 'Validation', 'Real'), 'output': 'data/processed_novo/validation/real'},
        {'name': 'Test Fake', 'input': os.path.join(BASE_INPUT_PATH, 'Test', 'Fake'), 'output': 'data/processed_novo/test/fake'},
        {'name': 'Test Real', 'input': os.path.join(BASE_INPUT_PATH, 'Test', 'Real'), 'output': 'data/processed_novo/test/real'},
    ]
    # ---------------------------------------------
    
    print("====== INICIANDO PROCESSAMENTO EM LOTE DO NOVO DATASET ======")
    all_success = True
    for job in JOBS:
        if not run_job(job['name'], job['input'], job['output']):
            all_success = False
            print("Abortando o processamento em lote devido a um erro.")
            break
            
    if all_success:
        print("\n====== PROCESSAMENTO EM LOTE CONCLUÍDO COM SUCESSO ======")
    else:
        print("\n====== PROCESSAMENTO EM LOTE FALHOU ======")