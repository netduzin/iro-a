# Importações organizadas por categoria
# Bibliotecas padrão
import streamlit as st

# Processamento de dados e ML
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Integração com modelos LLM
import ollama

# ==============================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ==============================================
st.set_page_config(
    page_title="iRoça Online",
    page_icon="🌱",
    layout="wide"
)

# ==============================================
# ESTILOS CSS PERSONALIZADOS
# ==============================================
# (Mantive seus estilos originais, apenas organizei visualmente)
css_styles = """
<style>
    /* Estilos gerais para mensagens do chat */
    .user-message {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border: 1px solid #d3d3d3;
    }
    .assistant-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border: 1px solid #bbdefb;
    }

    /* Container do chat */
    .chat-container {
        margin-bottom: 20px;
    }

    /* Cabeçalho e navegação */
    .header {
        background-color: #4d3319;
        padding: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .menu {
        display: flex;
        gap: 20px;
    }
    .menu button {
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .menu button:hover {
        text-decoration: underline;
        transform: translateY(-2px);
    }

    /* Botões de login/cadastro */
    .buttons {
        display: flex;
        gap: 10px;
    }
    .login, .sign {
        padding: 8px 15px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .login {
        background-color: #3d2b1f;
        color: white;
    }
    .sign {
        background-color: #28a745;
        color: white;
    }

    /* Seção de contato */
    .contact-section {
        display: flex;
        background-color: #f4f1e8;
        padding: 40px;
        border-radius: 10px;
        margin-top: 30px;
    }
    .contact-info {
        flex: 1;
        padding-right: 20px;
    }
    .contact-form {
        flex: 1;
        background-color: #d9c7a1;
        padding: 20px;
        border-radius: 10px;
    }
    .contact-form input, .contact-form textarea {
        width: 100%;
        padding: 10px;
        margin-top: 10px;
        border: none;
        border-radius: 5px;
    }
    .contact-form button {
        margin-top: 10px;
        background-color: #8b5e3b;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }

    /* Seção de previsões */
    .prediction-section {
        background-color: #e9f1f5;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .prediction-form {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
    }
    .prediction-form div {
        flex: 1;
        min-width: 280px;
    }
    .prediction-form select, .prediction-form input {
        width: 100%;
        padding: 10px;
        margin-top: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .prediction-form button {
        width: 100%;
        padding: 12px;
        margin-top: 15px;
        background-color: #28a745;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .prediction-form button:hover {
        background-color: #218838;
    }

    /* Seções genéricas */
    .section {
        padding: 40px 0;
    }
    .custom-container {
        background-color: #f4e1c1;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
</style>
"""

# Aplicando os estilos CSS
st.markdown(css_styles, unsafe_allow_html=True)

# ==============================================
# BARRA DE NAVEGAÇÃO SUPERIOR
# ==============================================
navigation_bar = """
<div class='header'>
    <div>
        <span style='color:white; font-size:24px; font-weight:bold;'>🌱 iRoça</span>
    </div>
    <div class='menu'>
        <button onclick="scrollToSection('previsoes')">🔮 Previsões</button>
        <button onclick="scrollToSection('chatbot')">🤖 ChatBot</button>
        <button onclick="scrollToSection('contato')">📞 Contato</button>
    </div>
    <div class='buttons'>
        <span class='login'>Entrar</span>
        <span class='sign'>Registrar</span>
    </div>
</div>
"""
st.markdown(navigation_bar, unsafe_allow_html=True)

# ==============================================
# SEÇÃO DE PREVISÕES AGRÍCOLAS
# ==============================================
st.markdown("""<div id='previsoes' class='section'></div>""", unsafe_allow_html=True)
st.markdown("""<h2>🔮 Previsões Agrícolas</h2>""", unsafe_allow_html=True)
st.write("Preencha os dados abaixo para prever a área colhida:")

# Dicionários para mapeamento (nomes amigáveis → valores do modelo)
MAPEAMENTO_PRODUTOS = {
    "Cana-de-Açúcar": "Cana_de_acucar",
    "Cereais (outros)": "cereais",
    "Soja": "soja",
    "Milho": "milho",
    "Laranja": "laranja"
}

MAPEAMENTO_MESES = {
    "Janeiro": "Janeiro",
    "Fevereiro": "fevereiro",
    "Março": "marco",
    "Abril": "abril",
    "Maio": "maio",
    "Junho": "junho",
    "Julho": "julho"
}

MAPEAMENTO_ESTADOS = {
    "Rondônia": "rondonia",
    "Acre": "acre",
    "Amazonas": "amazonas",
    "Roraima": "roraima",
    "Pará": "para",
    "Amapá": "amapa",
    "Tocantins": "tocantins",
    "Maranhão": "maranhao",
    "Piauí": "piaui",
    "Ceará": "ceara",
    "Rio Grande do Norte": "rio_grande_do_norte",
    "Paraíba": "paraiba",
    "Pernambuco": "pernambuco",
    "Alagoas": "alagoas",
    "Sergipe": "sergipe",
    "Bahia": "bahia",
    "Minas Gerais": "minas_gerais",
    "Espírito Santo": "espirito_santo",
    "Rio de Janeiro": "rio_de_janeiro",
    "São Paulo": "sao_paulo",
    "Paraná": "parana",
    "Santa Catarina": "santa_catarina",
    "Rio Grande do Sul": "rio_grande_do_sul",
    "Mato Grosso do Sul": "mato_grosso_do_sul",
    "Mato Grosso": "mato_grosso",
    "Goiás": "goias",
    "Distrito Federal": "distrito_federal"
}
# Carregamento do modelo - Se não encontrar, usa um modelo dummy
try:
    model = joblib.load('random_forest_model.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
except FileNotFoundError:
    st.warning("Modelo não encontrado. Usando modelo de demonstração.")

    # Criando um modelo dummy para demonstração
    model = RandomForestRegressor()

    # Encoders com valores de exemplo
    label_encoders = {
        'Produto': LabelEncoder().fit(['Cana_de_acucar', 'cereais', 'soja', 'milho', 'laranja']),
        'Mes': LabelEncoder().fit(['Janeiro', 'fevereiro', 'marco', 'abril', 'maio', 'junho', 'julho']),
        'estado': LabelEncoder().fit(list(MAPEAMENTO_ESTADOS.values()))
    }
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#

# Formulário de previsão em duas colunas
col1, col2 = st.columns(2)

with col1:
    # Mostra os nomes amigáveis, mas armazena os valores originais
    produto_nome = st.selectbox("Produto", list(MAPEAMENTO_PRODUTOS.keys()))
    produto = MAPEAMENTO_PRODUTOS[produto_nome]

    mes_nome = st.selectbox("Mês", list(MAPEAMENTO_MESES.keys()))
    mes = MAPEAMENTO_MESES[mes_nome]

    estado_nome = st.selectbox("Estado", list(MAPEAMENTO_ESTADOS.keys()))
    estado = MAPEAMENTO_ESTADOS[estado_nome]

with col2:
    area_plantada = st.number_input("Área Plantada (hectares)", min_value=0, value=1000)

# Botão de previsão
if st.button("Prever Área Colhida"):
    # Pré-processamento dos dados
    input_data = pd.DataFrame({
        'Produto': [produto],
        'Mes': [mes],
        'Area_Plantada': [area_plantada],
        'estado': [estado]
    })

    # Codificação das variáveis categóricas
    for column in ['Produto', 'Mes', 'estado']:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Realizando a previsão
    prediction = model.predict(input_data)

    # Exibição do resultado formatado
    st.markdown(
        f"""
        <div style="background-color: #d9c7a1; color: black; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
            <h3>Área colhida prevista para {produto_nome} em {estado_nome}</h3>
            <p style="font-size: 22px; font-weight: bold;">{prediction[0]:,.2f} hectares</p>
            <p style="font-size: 16px; color: black;">Mês de referência: {mes_nome}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
# ==============================================
# SEÇÃO DO CHATBOT
# ==============================================
st.markdown("""<div id='chatbot' class='section'></div>""", unsafe_allow_html=True)
st.markdown("""<h2>🤖 ChatBot</h2>""", unsafe_allow_html=True)
st.write("Converse com nosso assistente virtual!")

# Inicialização do histórico de mensagens
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Campo de entrada do usuário
user_input = st.text_input("Digite sua pergunta:")


def obter_resposta(pergunta):
    """
    Obtém resposta do modelo LLM considerando o histórico da conversa.

    Args:
        pergunta (str): Pergunta do usuário

    Returns:
        dict: Resposta do modelo contendo a mensagem gerada
    """
    # Prepara o histórico no formato esperado pelo modelo
    mensagens = []
    for autor, mensagem in st.session_state.chat_history:
        role = "user" if autor == "Usuário" else "assistant"
        mensagens.append({"role": role, "content": mensagem})

    # Adiciona a nova pergunta com instrução para responder em português
    mensagens.append({"role": "user", "content": f"Responda sempre em português. {pergunta}"})

    # Chama o modelo LLM
    return ollama.chat(
        model='gemma:2b',
        messages=mensagens
    )


# Botões do chat em colunas
col1, col2 = st.columns([0.4, 3])

with col1:
    if st.button("❓ Perguntar", key="btn_perguntar"):
        pergunta = user_input.strip()
        if pergunta:
            try:
                resposta = obter_resposta(pergunta)
                conteudo_resposta = resposta['message']['content']

                # Armazena a conversa no histórico
                st.session_state.chat_history.append(("Usuário", pergunta))
                st.session_state.chat_history.append(("Assistente", conteudo_resposta))

            except Exception as e:
                st.error("❌ Ocorreu um erro ao tentar se comunicar com o modelo.")
                st.exception(e)
        else:
            st.warning("⚠️ Por favor, digite uma pergunta antes de enviar.")

with col2:
    if st.button("🧹 Limpar conversa", key="btn_limpar"):
        st.session_state.chat_history = []

# Exibição do histórico de conversa
if 'chat_history' in st.session_state and st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for autor, mensagem in st.session_state.chat_history:
        if autor == "Usuário":
            st.markdown(f"""
            <div class='user-message'>
                <strong>{autor}:</strong> {mensagem}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='assistant-message'>
                <strong>{autor}:</strong> {mensagem}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================
# SEÇÃO DE CONTATO
# ==============================================
st.markdown("""<div id='contato' class='section'></div>""", unsafe_allow_html=True)
st.markdown("""<h2>📞 Contato</h2>""", unsafe_allow_html=True)

contact_section = """
<div class='contact-section'>
    <div class='contact-info'>
        <h3>📍 Localização</h3>
        <p>Jardim Brasil, 17800-000</p>
        <p>Adamantina - SP</p>
        <h3>📱 Siga-nos</h3>
        <p>📘 Facebook | 🐦 Twitter | 📷 Instagram | 📧 Email</p>
        <p style="margin-top: 50px;">&copy; 2025 Política de Privacidade</p>
    </div>
    <div class='contact-form'>
        <h3>✉️ Formulário de Contato</h3>
        <input type='text' placeholder='Seu Nome'>
        <input type='email' placeholder='Seu Email'>
        <textarea placeholder='Sua Mensagem' rows='4'></textarea>
        <button>Enviar Mensagem</button>
    </div>
</div>
"""
st.markdown(contact_section, unsafe_allow_html=True)