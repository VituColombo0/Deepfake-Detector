

import os
import hashlib
import time
from PIL import Image
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from github import Github, RateLimitExceededException




SEARCH_QUERIES = [
    "deepfake dataset sample",
    "faceforensics dataset",
    "celeb-df sample"
]


OUTPUT_FOLDER = 'data/unlabeled_images'

HASH_FILE = 'data/image_hashes.txt'


def calculate_hash(image_path):
    """Calcula o hash de uma imagem para identificá-la."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((128, 128))
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None

def load_existing_hashes():
    """Carrega os hashes de imagens que já temos."""
    if not os.path.exists(HASH_FILE):
        return set()
    with open(HASH_FILE, 'r') as f:
        return set(f.read().splitlines())

def save_new_hash(new_hash):
    """Adiciona um novo hash ao nosso banco de dados."""
    with open(HASH_FILE, 'a') as f:
        f.write(new_hash + '\n')

def scrape_repo_page(repo_url, existing_hashes):
    """Raspa uma página de repositório para encontrar e baixar imagens novas."""
    print(f"  ⛏️  Minerando repositório: {repo_url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    new_count, duplicate_count = 0, 0
    try:
        response = requests.get(repo_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for link in soup.find_all('a', class_='Link--primary'):
            href = link.get('href')
            if href and (href.endswith(('.png', '.jpg', '.jpeg'))):
                raw_url = urljoin("https://raw.githubusercontent.com", href.replace('/blob', ''))
                img_name = os.path.basename(raw_url).split('?')[0]
                img_path = os.path.join(OUTPUT_FOLDER, img_name)
                
                img_data = requests.get(raw_url, headers=headers).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                img_hash = calculate_hash(img_path)
                if img_hash:
                    if img_hash in existing_hashes:
                        os.remove(img_path)
                        duplicate_count += 1
                    else:
                        existing_hashes.add(img_hash)
                        save_new_hash(img_hash)
                        new_count += 1
        
        print(f"    -> Concluído. Novas imagens: {new_count} | Duplicatas ignoradas: {duplicate_count}")
        return new_count
    except Exception as e:
        print(f"    -> [AVISO] Falha ao minerar {repo_url}. Erro: {e}")
        return 0

def main():
    """Função principal que orquestra a descoberta e a mineração."""
    print("--- INICIANDO PROTOCOLO GARIMPEIRO (MÓDULO GITHUB) ---")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    existing_hashes = load_existing_hashes()
    
    print(f"\n📡 Etapa 1: Usando o 'Radar' para descobrir fontes...")
    github_client = Github() 
    discovered_repos = set()

    for query in SEARCH_QUERIES:
        print(f"  Buscando por: '{query}'...")
        try:
            repositories = github_client.search_repositories(query=query)
            
            for repo in repositories[:5]:
                
                discovered_repos.add(repo.html_url)
                discovered_repos.add(f"{repo.html_url}/tree/main/samples")
                discovered_repos.add(f"{repo.html_url}/tree/master/samples")
                discovered_repos.add(f"{repo.html_url}/tree/main/images")
                discovered_repos.add(f"{repo.html_url}/tree/master/images")

        except RateLimitExceededException:
            print("  [AVISO] Limite de busca da API do GitHub atingido. Continuando com o que foi encontrado.")
            break
        except Exception as e:
            print(f"  [ERRO] Erro na busca: {e}")
        time.sleep(10) 

    if not discovered_repos:
        print("Radar não encontrou nenhuma fonte nova.")
        return

    print(f"\n✅ Radar concluído. {len(discovered_repos)} URLs em potencial encontradas.")
    print("---")
    print(f"⛏️  Etapa 2: Iniciando a mineração em cada fonte...")
    
    total_new_images = 0
    for repo_url in discovered_repos:
        total_new_images += scrape_repo_page(repo_url, existing_hashes)
        time.sleep(2) 

    print("\n--- RELATÓRIO FINAL DA MINERAÇÃO ---")
    print(f"Total de novas imagens únicas coletadas nesta execução: {total_new_images}")
    print(f"Novas imagens estão prontas para rotulagem em: '{OUTPUT_FOLDER}'")

if __name__ == '__main__':
    main()