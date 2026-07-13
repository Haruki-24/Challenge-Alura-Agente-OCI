import os
import streamlit as st
from rag_agent_str import grafo_operativo 

# --- Función para cargar documentos dinámicamente ---
def obtener_documentos_activos(ruta_carpeta="data"):
    # Si la carpeta no existe, devolvemos una lista vacía
    if not os.path.exists(ruta_carpeta):
        return []
    
    # Listar solo archivos con extensión .pdf
    archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.pdf')]
    
    return sorted(archivos) # Devuelve la lista ordenada alfabéticamente

# --- Configuración de la Página ---
# Cambiamos a layout="wide" para dar espacio a la barra lateral derecha
st.set_page_config(page_title="Agente OCI", page_icon="🛢️", layout="wide")

# --- Inicialización del Estado de la Sesión ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Buen día, soy el Asistente operativo de OCI. ¿Cómo puedo ayudarte?"}
    ]

# --- Estructura de Diseño (Columnas) ---
# Creamos la columna principal para el chat y la columna derecha para la info
col_chat, col_info = st.columns([3, 1], gap="large")

# --- BARRA LATERAL DERECHA (Panel de Información) ---
with col_info:
    st.markdown("### 📂 Documentos Cargados")
    st.caption("Base de conocimiento activa del agente:")
    
    # OBTENCIÓN DINÁMICA: Escanea la carpeta 'data' en cada recarga de la app
    documentos = obtener_documentos_activos("data")
    
    if documentos:
        for doc in documentos:
            st.markdown(f"- `{doc}`")
    else:
        st.warning("⚠️ No se encontraron documentos en la carpeta 'data'.")
        
    st.markdown("---")
    
    st.markdown("### 💡 Consultas Sugeridas")
    st.caption("Haz clic en cualquier sugerencia para consultar al agente:")
    
    # Botones interactivos para las sugerencias
    sug_mantenimiento = st.button("¿Cuál es el programa de mantenimiento?")
    sug_seguridad = st.button("¿Cuál es el procedimiento de seguridad ante derrames?")
    sug_epp = st.button("¿Cuál es el inventario actual de EPP?")

# --- CAPTURA DE ENTRADA (Chat Input & Botones) ---
prompt = st.chat_input("Escribe tu consulta sobre operaciones...")

# Si el usuario hace clic en una sugerencia, la asignamos como el prompt activo
if sug_mantenimiento:
    prompt = "¿Cuál es el programa de mantenimiento?"
elif sug_seguridad:
    prompt = "¿Cuál es el procedimiento de seguridad ante derrames?"
elif sug_epp:
    prompt = "¿Cuál es el inventario actual de EPP?"

# --- COLUMNA PRINCIPAL (Interfaz de Chat) ---
with col_chat:
    st.title(" Agente OCI - Control de Operaciones")
    st.caption("Tu asistente inteligente para la coordinación de campo.")
    st.markdown("---")

    # Mostrar el historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Procesamiento de la Consulta ---
    if prompt:
        # 1. Registrar y mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Llamar al agente LangGraph
        input_state = {"pregunta": prompt}
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(" Procesando tu consulta...")

            try:
                # Ejecutar grafo 
                estado_final = grafo_operativo.invoke(input_state)
                
                # Extraer respuesta
                respuesta = estado_final.get("respuesta_RAG", "Lo siento, no pude generar una respuesta.")
                
                # Mostrar citaciones o el resultado del triaje
                with st.expander("Ver detalles del análisis"):
                    st.json({
                        "Triaje": estado_final.get("triaje"),
                        "Acción Final": estado_final.get("accion_final"),
                        "Éxito del RAG": estado_final.get("rag_exito"),
                        "Citaciones": estado_final.get("citaciones", [])
                    })

            except Exception as e:
                placeholder.error(f"Error: {e}")
                respuesta = f"Ocurrió un error inesperado: {e}"

            # Respuesta final
            placeholder.markdown(respuesta)
        
        # 3. Guardar la respuesta del asistente en el historial
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
