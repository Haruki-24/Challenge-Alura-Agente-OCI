import streamlit as st
from rag_agent_str import grafo_operativo 

# --- Configuración de la Página ---
st.set_page_config(page_title="Agente OCI", page_icon="🛢️", layout="centered")
st.title(" Agente OCI - Control de Operaciones")
st.caption("Tu asistente inteligente para la coordinación de campo.")

# --- Inicialización del Estado de la Sesión ---

if "messages" not in st.session_state:
    # Inicializa el historial con mensaje de bienvenida
    st.session_state.messages = [
        {"role": "assistant", "content": "Buen día, soy el Asistente operativo de OCI. ¿Como puedo ayudarte?"}
    ]

# --- Historial de Mensajes ---
# Recorre todos los mensajes guardados en la sesión y los muestra en el chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Capturar Entrada del Usuario ---
if prompt := st.chat_input("Escribe tu consulta sobre operaciones..."):
    
    # 1. Mensaje del usuario al historial y lo muestra
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Llamar al agente LangGraph
    # Preparar estado inicial de grafo
    input_state = {"pregunta": prompt}
    
    with st.chat_message("assistant"):
        # Placeholder para simular el "pensamiento" del agente
        placeholder = st.empty()
        placeholder.markdown(" Procesando tu consulta...")

        try:
            # Ejecutar grafo. 
            estado_final = grafo_operativo.invoke(input_state)
            
            # Extrae respuesta final de estado
            respuesta = estado_final.get("respuesta_RAG", "Lo siento, no pude generar una respuesta.")
            
      
            # Mostrar citaciones o el resultado del triaje
            with st.expander("🔍 Ver detalles del análisis"):
                st.json({
                    "Triaje": estado_final.get("triaje"),
                    "Acción Final": estado_final.get("accion_final"),
                    "Éxito del RAG": estado_final.get("rag_exito"),
                    "Citaciones": estado_final.get("citaciones", [])
                })

        except Exception as e:
            placeholder.error(f"Error: {e}")
            respuesta = f" Ocurrió un error inesperado: {e}"

        # Respuesta final
        placeholder.markdown(respuesta)
    
    # 3. Añade la respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": respuesta})

