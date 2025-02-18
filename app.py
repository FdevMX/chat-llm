import streamlit as st
import time
from groq import Groq
from typing import Generator

st.title("Chat LLM")

# Declaramos la api de Groq
client = Groq(
    api_key = st.secrets["ngroqAPIKey"],
)

# Listar los modelos disponibles
modelos = ['llama3-70b-8192', 'deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-qwen-32b', 'qwen-2.5-coder-32b', 'mixtral-8x7b-32768', 'gemma2-9b-it']

def generate_chat_response(chat_completion) -> Generator[str, None, None]:
    """Generate responses from the chat completion.
       Genera respuestas a partir de la informacion de completado de chat, mostrando caracter por caracter.

    Args:
        chat_completion (str): La informacion de completado de chat.

    Yields:
        str: Cada respuesta generada.
    """
    
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            
# Inicializamos el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []   

# Muestra mensaje de chat desde la historia en la aplicacion cada vez que la aplicacion se ejecuta
with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
# Mostramos la lista de modelos en el sidebar
parModelo = st.sidebar.selectbox('Modelos', options=modelos, index=0)

# Mostramos el campo para el promt
prompt = st.chat_input("Escribe un mensaje...")

if prompt:
    # Mostrar mensaje del usuario en el contenedor de mensajes de chat
    st.chat_message("user").markdown(prompt)
    # Agregar mensaje del usuario al historial de mensajes
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        chat_completion = client.chat.completions.create(
            model = parModelo,
            messages = [
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ], # Entregamos historial de chat para que el modelo tenga memoria
            stream = True
        )
        
        # Mostrar respuestas del modelo en el contenedor de mensajes de chat
        with st.chat_message("assistant"):
            chat_responses_generator = generate_chat_response(chat_completion)
            # Usamos st.write_stream para simular escritura de caracteres
            full_response = st.write_stream(chat_responses_generator)
        # Agregar respuestas del modelo al historial de mensajes
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})