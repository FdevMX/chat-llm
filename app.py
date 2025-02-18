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
            # Si el mensaje del asistente viene como diccionario, mostramos expander y respuesta final
            if message["role"] == "assistant" and isinstance(message["content"], dict):
                if message["content"].get("thinking"):
                    with st.expander("Mostrar pensamiento"):
                        st.write(message["content"]["thinking"])
                st.markdown(message["content"]["final"])
            else:
                st.markdown(message["content"])
            
# Mostramos la lista de modelos en el sidebar
parModelo = st.sidebar.selectbox('Modelos', options=modelos, index=0)

# Mostramos el campo para el promt
prompt = st.chat_input("Escribe un mensaje...")

if prompt:
    # Mostrar mensaje del usuario y guardarlo en el historial
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        # Transforma el historial para enviar solo strings como contenido
        messages_to_send = [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else m["content"]["final"]}
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
            
            status_ph.empty()  # Eliminamos el mensaje "Pensando..."
            
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
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})