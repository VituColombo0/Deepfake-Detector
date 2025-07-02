// script.js (VERSÃO FINAL E COMPLETA)

// Adiciona todos os event listeners quando o documento HTML estiver pronto.
document.addEventListener("DOMContentLoaded", function() {
    
    // Configura o formulário de Análise
    const analysisForm = document.getElementById('upload-form-analysis');
    if (analysisForm) {
        analysisForm.addEventListener('submit', function(event) { 
            // A linha mais importante: impede que a página recarregue ao enviar.
            event.preventDefault(); 
            
            const fileInput = document.getElementById('file-input-analysis');
            const file = fileInput.files[0];

            if (file) {
                // Verifica se o arquivo é um vídeo ou imagem e chama a função correta
                const type = file.type.startsWith('video/') ? 'video' : 'image';
                handleAnalysis(type, file);
            } else {
                // Mostra uma mensagem se nenhum ficheiro foi selecionado
                const resultText = document.getElementById('result-text');
                const resultContainer = document.getElementById('result-container');
                resultText.innerHTML = `<span class="fake">Por favor, selecione um ficheiro primeiro.</span>`;
                resultContainer.classList.remove('hidden');
            }
        });
    }
    
    // Configura os botões da aba de curadoria
    const nextImageBtn = document.getElementById('next-image-button');
    if (nextImageBtn) {
        nextImageBtn.addEventListener('click', fetchUnlabeledImage);
        document.getElementById('real-button').addEventListener('click', () => labelImage('real'));
        document.getElementById('fake-button').addEventListener('click', () => labelImage('fake'));
    }
    
    // Inicia com a primeira aba e busca o status do modelo
    document.getElementsByClassName("tab-button")[0].click();
    fetchModelStatus();
});

// Função para gerenciar a troca de abas
function openTab(evt, tabName) {
    let i, tabcontent, tabbuttons;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tabbuttons = document.getElementsByClassName("tab-button");
    for (i = 0; i < tabbuttons.length; i++) {
        tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";

    // Carrega a primeira imagem da curadoria se a aba for aberta pela primeira vez
    if (tabName === 'CurationTab' && !document.getElementById('curation-image-area').querySelector('img')) {
        fetchUnlabeledImage();
    }
}

// Função para buscar e exibir o status do modelo
async function fetchModelStatus() {
    const statusPanel = document.getElementById('status-panel');
    try {
        const response = await fetch('http://127.0.0.1:5000/status');
        const data = await response.json();
        if (response.ok) {
            statusPanel.innerHTML = `<span>✅ <strong>Modelo Ativo:</strong> ${data.model_name}</span><span>⚙️ <strong>Parâmetros Treináveis:</strong> ${data.trainable_parameters}</span>`;
        } else {
            statusPanel.innerHTML = `<span>❌ Erro ao carregar status: ${data.error}</span>`;
        }
    } catch (error) {
        statusPanel.innerHTML = `<span>🔌 Erro de conexão com o servidor.</span>`;
    }
}

// --- LÓGICA DE ANÁLISE ---
async function handleAnalysis(type, file) {
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');
    const xaiGallery = document.getElementById('xai-gallery');
    
    const formData = new FormData();
    formData.append('file', file);

    loader.classList.remove('hidden');
    resultText.innerHTML = `Analisando ${type}... Isso pode levar vários minutos para vídeos.`;
    xaiGallery.innerHTML = "";
    resultContainer.classList.remove('hidden');
    
    const endpoint = type === 'image' ? '/predict' : '/predict_video';
    const fullUrl = `http://127.0.0.1:5000${endpoint}`;

    try {
        const response = await fetch(fullUrl, { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Erro no servidor.');

        if (type === 'image') {
            loader.classList.add('hidden');
            displayImageResults(data);
        } else { // type === 'video'
            pollForResult(data.job_id);
        }
    } catch (error) {
        loader.classList.add('hidden');
        resultText.innerHTML = `<span class="fake">Erro ao iniciar a análise: ${error.message}</span>`;
    }
}

function pollForResult(jobId) {
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');
    const intervalId = setInterval(async () => {
        try {
            const statusResponse = await fetch(`http://127.0.0.1:5000/results/${jobId}`);
            if (!statusResponse.ok) return;
            const statusData = await statusResponse.json();
            if (statusData.status === 'complete' || statusData.status === 'error') {
                clearInterval(intervalId);
                loader.classList.add('hidden');
                if (statusData.status === 'complete') {
                    displayVideoResults(statusData.result);
                } else {
                    resultText.innerHTML = `<span class="fake">Erro na análise do vídeo: ${statusData.result.error}</span>`;
                }
            }
        } catch (e) {
            clearInterval(intervalId);
            loader.classList.add('hidden');
            resultText.innerHTML = `<span class="fake">Erro de conexão ao verificar status.</span>`;
        }
    }, 3000);
}

function displayImageResults(data) {
    const resultText = document.getElementById('result-text');
    const xaiGallery = document.getElementById('xai-gallery');
    resultText.innerHTML = (data.verdict === 'REAL') ? `<span class="real">Veredito: REAL</span><br>Confiança: ${data.confidence}` : `<span class="fake">Veredito: FAKE</span><br>Confiança: ${data.confidence}`;
    if (data.original_face_b64) {
        xaiGallery.innerHTML = `
            <h3>Explicação Visual (Grad-CAM)</h3>
            <div class="gallery-item"><img src="data:image/jpeg;base64,${data.original_face_b64}" alt="Rosto Detectado"><p>Rosto Detectado</p></div>
            <div class="gallery-item"><img src="data:image/jpeg;base64,${data.heatmap_b64}" alt="Mapa de Calor"><p>Foco da IA</p></div>
            <div class="gallery-item"><img src="data:image/jpeg;base64,${data.overlaid_b64}" alt="Análise Sobreposta"><p>Análise Sobreposta</p></div>`;
    }
}

function displayVideoResults(data) {
    const resultText = document.getElementById('result-text');
    const xaiGallery = document.getElementById('xai-gallery'); 
    resultText.innerHTML = (data.verdict === 'REAL') ? `<span class="real">Veredito: REAL</span><br>Confiança: ${data.confidence}` : `<span class="fake">Veredito: FAKE</span><br>Confiança: ${data.confidence}`;
    if (data.xai_frames && data.xai_frames.length > 0) {
        xaiGallery.innerHTML = `<h3>Análise Detalhada dos Frames Mais Suspeitos</h3>`;
        data.xai_frames.forEach((frame, index) => {
            const card = document.createElement('div');
            card.className = 'gallery-item';
            card.innerHTML = `<h4>Frame Suspeito #${index + 1}</h4><img src="data:image/jpeg;base64,${frame.original_b64}" alt="Frame Original"><p>Análise XAI</p><img src="data:image/jpeg;base64,${frame.xai_b64}" alt="Frame com XAI">`;
            xaiGallery.appendChild(card);
        });
    }
}

// --- LÓGICA DA CURADORIA ---
let currentImagePath = null;
async function fetchUnlabeledImage() {
    const curationArea = document.getElementById('curation-image-area');
    const curationButtons = document.getElementById('curation-buttons');
    curationArea.innerHTML = `<div class="loader"></div>`;
    curationButtons.classList.add('hidden');
    try {
        const response = await fetch('http://127.0.0.1:5000/get_unlabeled_image');
        const data = await response.json();
        if (response.ok) {
            curationArea.innerHTML = `<img src="data:image/jpeg;base64,${data.image_b64}" alt="Imagem para rotular">`;
            currentImagePath = data.path;
            curationButtons.classList.remove('hidden');
        } else {
            curationArea.innerHTML = `<p style="color: #ffc107;">${data.error}</p>`;
        }
    } catch (e) {
        curationArea.innerHTML = `<p class="fake">Erro de conexão ao buscar imagem.</p>`;
    }
}
async function labelImage(label) {
    if (!currentImagePath) return;
    document.getElementById('real-button').disabled = true;
    document.getElementById('fake-button').disabled = true;
    try {
        await fetch('http://127.0.0.1:5000/label_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: currentImagePath, label: label })
        });
        await fetchUnlabeledImage();
    } catch (e) {
        alert('Erro de conexão ao rotular imagem.');
    } finally {
        document.getElementById('real-button').disabled = false;
        document.getElementById('fake-button').disabled = false;
    }
}