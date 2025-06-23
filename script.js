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
}

// Inicia com a primeira aba ativa
document.getElementsByClassName("tab-button")[0].click();


// Lógica para o formulário de IMAGEM
document.getElementById('upload-form-image').addEventListener('submit', async function(event) {
    event.preventDefault();
    handleUpload('image');
});

// Lógica para o formulário de VÍDEO
document.getElementById('upload-form-video').addEventListener('submit', async function(event) {
    event.preventDefault();
    handleUpload('video');
});


async function handleUpload(type) {
    const fileInput = document.getElementById(type === 'image' ? 'file-input-image' : 'file-input-video');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');

    if (fileInput.files.length === 0) {
        resultText.innerHTML = "Por favor, selecione um arquivo.";
        resultContainer.classList.remove('hidden');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Mostra o loader e a mensagem de análise
    loader.classList.remove('hidden');
    resultText.innerHTML = `Analisando ${type === 'image' ? 'imagem' : 'vídeo'}... Isso pode levar alguns instantes.`;
    resultContainer.classList.remove('hidden');

    const endpoint = type === 'image' ? 'http://127.0.0.1:5000/predict' : 'http://127.0.0.1:5000/predict_video';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        loader.classList.add('hidden'); // Esconde o loader

        if (response.ok) {
            if (data.verdict === 'REAL') {
                resultText.innerHTML = `<span class="real">Veredito: REAL</span><br>Confiança: ${data.confidence}`;
            } else {
                resultText.innerHTML = `<span class="fake">Veredito: FAKE</span><br>Confiança: ${data.confidence}`;
            }
        } else {
            resultText.innerHTML = `<span class="fake">Erro: ${data.error}</span>`;
        }
    } catch (error) {
        loader.classList.add('hidden');
        resultText.innerHTML = `<span class="fake">Erro de conexão. O servidor Python (api.py) está rodando?</span>`;
    }
}