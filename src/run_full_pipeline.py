# src/run_full_pipeline.py

import subprocess
import sys
import time
import os

# --- ORDEM DAS TAREFAS DO PIPELINE ---
# Garanta que os nomes dos scripts abaixo correspondem aos seus arquivos em src/
PIPELINE_STEPS = [
    "process_all_datasets.py",
    "prepare_final_dataset.py",
    "train_final_model.py" # O nosso script de treino mais recente
]
# ------------------------------------

def run_step(script_name):
    """Executa um passo do pipeline e verifica se houve erro."""
    
    script_path = os.path.join("src", script_name)
    if not os.path.exists(script_path):
        print(f"\n[ERRO FATAL] O script '{script_path}' não foi encontrado!")
        return False

    command = [sys.executable, script_path] # sys.executable garante que usamos o python do ambiente conda
    
    print(f"\n{'='*25}\nINICIANDO ETAPA: {script_name}\n{'='*25}")
    
    try:
        # Executa o script e aguarda sua conclusão
        subprocess.run(command, check=True)
        print(f"\n--- ETAPA '{script_name}' CONCLUÍDA COM SUCESSO ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO FATAL] A ETAPA '{script_name}' FALHOU COM CÓDIGO DE SAÍDA {e.returncode}.")
        return False
    except KeyboardInterrupt:
        print(f"\n[AVISO] Pipeline interrompido pelo usuário.")
        return False


if __name__ == '__main__':
    start_time = time.time()
    print("<<<<< INICIANDO PIPELINE COMPLETO DE CRIAÇÃO DE MODELO >>>>>")
    
    # Executa cada passo em sequência
    for script_file in PIPELINE_STEPS:
        if not run_step(script_file):
            print("\n<<<<< PIPELINE INTERROMPIDO DEVIDO A UM ERRO >>>>>")
            break
    else: # Este 'else' pertence ao 'for', só executa se o loop terminar sem 'break'
        end_time = time.time()
        total_duration_hours = (end_time - start_time) / 3600
        print(f"\n<<<<< PIPELINE COMPLETO EXECUTADO COM SUCESSO EM {total_duration_hours:.2f} HORAS >>>>>")
        print("UM NOVO MODELO DE IA FOI GERADO!")