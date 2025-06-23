import streamlit as st

st.set_page_config(page_title="Teste de CSS")

st.title("Teste de Renderização")

st.write("Se o teste funcionar, a aparência desta página será alterada.")
st.markdown("---")

# --- Teste de Diagnóstico ---
# Vamos tentar injetar o CSS mais simples possível.
# Se isso não funcionar, nada mais vai.

st.markdown("""
<style>
/* Este CSS deveria mudar o fundo de toda a página para um cinza escuro */
.main .block-container {
    background-color: #1E1E1E;
}

/* Este CSS deveria mudar a cor do título principal para azul */
h1 {
    color: #00A9FF;
}
</style>
""", unsafe_allow_html=True)

st.success("O script rodou sem erros.")
st.info("Por favor, verifique se o fundo da página está escuro e o título 'Teste de Renderização' está azul.")