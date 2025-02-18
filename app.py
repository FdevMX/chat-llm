import streamlit as st
import time
from groq import Groq
from typing import Generator
from PIL import Image

logo_img = Image.open("logo/groq.png")
st.set_page_config(page_title='Groq LLM', page_icon=logo_img)

# Declaramos la api de Groq
client = Groq(
    api_key=st.secrets["ngroqAPIKey"],
)

# Listar los modelos disponibles
modelos = ['llama3-70b-8192', 'deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-qwen-32b',
           'qwen-2.5-coder-32b', 'mixtral-8x7b-32768', 'gemma2-9b-it']

# Mapear cada modelo a su logo respectivo
model_logos = {
    'llama3-70b-8192': 'logo/ollama.png',
    'deepseek-r1-distill-llama-70b': 'logo/deepseek.png',
    'deepseek-r1-distill-qwen-32b': 'logo/deepseek.png',
    'qwen-2.5-coder-32b': 'logo/qwen.png',
    'mixtral-8x7b-32768': 'logo/mistral.png',
    'gemma2-9b-it': 'logo/gemini.png'
}

def format_model_name(model_key: str) -> str:
    return model_key.split("/")[-1]

# Usa el modelo seleccionado (o el valor por defecto) y formatea el nombre para mostrar
selected_model = st.session_state.get('modelo_select', modelos[0])
model_display_name = format_model_name(selected_model)
# Seleccionar el logo en función del modelo seleccionado
logo_model = model_logos.get(selected_model)
if logo_model:
    st.image(logo_model, width=80)  # Ajusta el ancho según lo necesites
st.header(f"`{model_display_name}` model")

def generate_chat_response(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Inicializamos el historial de chat actual y el historial de conversaciones
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversations" not in st.session_state:
    st.session_state.conversations = {}  # dict {nombre: conversación}
    
if "current_conv" not in st.session_state:
    st.session_state.current_conv = "Conversación actual"

# --- Sidebar ---
# Agrega el logo justo al inicio del sidebar
current_theme = st.get_option("theme.base")
if current_theme == "dark":
    logo_path = "logo/light.png"
else:
    logo_path = "logo/dark.png"
st.sidebar.image(logo_path, width=150)
# st.sidebar.markdown("<h1 style='font-size:26px; color: yellow;'>🔍 Groq LLM Chat</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:15px; color:grey;'>A free AI chatbot</p>", unsafe_allow_html=True)

# Ahora el selector de modelos
parModelo = st.sidebar.selectbox('Modelos', options=modelos, index=0, key="modelo_select")

# Detectar cambio de modelo y agregar mensaje de sistema (sin ícono) solo si el chat ya tiene contenido
if "prev_model" not in st.session_state:
    st.session_state.prev_model = parModelo
elif parModelo != st.session_state.prev_model:
    if st.session_state.messages:  # Si hay contenido en el chat
        # Si el último mensaje es de cambio de modelo, se reemplaza su contenido
        if st.session_state.messages[-1]["role"] == "system" and st.session_state.messages[-1]["content"].startswith("Modelo cambiado:"):
            st.session_state.messages[-1]["content"] = f"`Modelo cambiado, se está usando ahora {parModelo}`"
        else:
            st.session_state.messages.append({"role": "system", "content": f"`Modelo cambiado, se está usando ahora {parModelo}`"})
    st.session_state.prev_model = parModelo

if st.sidebar.button("Nueva conversación"):
    if st.session_state.messages:
        title = None
        for m in st.session_state.messages:
            if m["role"] == "user":
                title = m["content"]
                break
        if title:
            # Genera un título único usando parte del contenido y la hora
            title = f"{title.strip()[:20]} - {time.strftime('%H:%M:%S')}"
        else:
            title = f"Conv. - {time.strftime('%H:%M:%S')}"
        st.session_state.conversations[title] = st.session_state.messages.copy()
    st.session_state.messages = []
    st.session_state.current_conv = "Conversación actual"
    
conv_options = list(st.session_state.conversations.keys())
conv_options.insert(0, st.session_state.current_conv)  # Conversación actual al frente
selected_conv = st.sidebar.selectbox("Historial de chats", conv_options)

# ------------------------------------------------------------------

# Mostrar el historial
if selected_conv != st.session_state.current_conv:
    st.write(f"Mostrando historial de: {selected_conv}")
    chat_history = st.session_state.conversations[selected_conv]
else:
    chat_history = st.session_state.messages

# Renderizar el chat...
with st.container():
    for message in chat_history:
        if message["role"] == "system":
            st.write(message["content"])
        else:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and isinstance(message["content"], dict):
                    if message["content"].get("thinking"):
                        with st.expander("Mostrar pensamiento"):
                            st.write(message["content"]["thinking"])
                    st.markdown(message["content"]["final"])
                else:
                    st.markdown(message["content"])

# Campo para el prompt
prompt = st.chat_input("Escribe un mensaje...")

if prompt:
    # Si se selecciona una conversación guardada, reanudarla (haciendo copia)
    if selected_conv != st.session_state.current_conv:
        st.session_state.current_conv = selected_conv
        st.session_state.messages = st.session_state.conversations[selected_conv].copy()
        
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        messages_to_send = [
            {"role": m["role"], 
             "content": m["content"] if isinstance(m["content"], str)
                        else m["content"]["final"]}
            for m in st.session_state.messages
        ]
        
        chat_completion = client.chat.completions.create(
            model=parModelo,
            messages=messages_to_send,
            stream=True
        )
        
        if parModelo in ['deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-qwen-32b']:
            status_ph = st.empty()
            status_ph.text("Pensando...")
            full_response = ""
            for chunk in chat_completion:
                piece = chunk.choices[0].delta.content or ""
                full_response += piece
            status_ph.empty()
            
            if "<think>" in full_response and "</think>" in full_response:
                start = full_response.find("<think>")
                end = full_response.find("</think>")
                thinking_text = full_response[start+len("<think>"):end].strip()
                final_answer = (full_response[:start] + full_response[end+len("</think>"):]).strip()
            else:
                thinking_text = ""
                final_answer = full_response
            
            with st.chat_message("assistant"):
                if thinking_text:
                    with st.expander("Mostrar pensamiento", expanded=True):
                        st.write(thinking_text)
                st.markdown(final_answer)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"thinking": thinking_text, "final": final_answer}
            })
        else:
            with st.chat_message("assistant"):
                full_response = st.write_stream(generate_chat_response(chat_completion))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Si no se está en la conversación "Conversación actual", actualizar el registro
        if st.session_state.current_conv != "Conversación actual":
            st.session_state.conversations[st.session_state.current_conv] = st.session_state.messages.copy()
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})