import os
import streamlit as st
import datetime
from langchain_core.messages import HumanMessage, AIMessage
# Importación del cerebro adaptado con el nuevo nombre
from rag_agent_gcp import grafo_operativo 

# --- CONFIGURACIÓN BASE DE LA PÁGINA ---
st.set_page_config(
    page_title="PETRO-ASSIST | Agente Operativo",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DETECCIÓN DINÁMICA DE DOCUMENTOS ---
def obtener_documentos_activos(ruta_carpeta="data"):
    if not os.path.exists(ruta_carpeta):
        return []
    # Escaneamos PDFs y Excels del repositorio
    archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.pdf') or f.endswith('.xlsx')]
    return sorted(archivos)

# --- INICIALIZACIÓN DEL HISTORIAL Y MOCK DATOS ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bienvenido Juan, ¿en qué puedo ayudarte hoy?"}
    ]

# Inyección CSS avanzada para transformar Streamlit en el panel de control PETRO-ASSIST
st.markdown("""
<style>
    /* Importación de fuentes industriales/tecnológicas */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600;700&family=Roboto+Mono:wght@400;500&display=swap');
    
    /* Configuración Global de la Aplicación */
    .stApp {
        background-color: #080D14 !important;
        background-image: radial-gradient(circle, #0c1622 10%, #080d14 90%) !important;
        color: #C0C0C0 !important;
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    /* Ocultar barra de herramientas nativa de Streamlit */
    header, footer, [data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0px !important;
    }
    
    /* Títulos e Headers */
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        color: #C0C0C0 !important;
        letter-spacing: 1px;
    }
    
    /* Tarjetas y Contenedores Generales */
    .petro-card {
        background: linear-gradient(145deg, #0f1621, #151d29);
        border: 1px solid #3A5A70;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* Tarjetas de Rol Activas/Inactivas */
    .role-card-active {
        background: rgba(0, 229, 255, 0.05);
        border: 1px solid #00E5FF;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 8px;
        box-shadow: 0 0 8px rgba(0, 229, 255, 0.1);
    }
    
    .role-card-inactive {
        background: rgba(15, 22, 33, 0.6);
        border: 1px solid #2B2B2B;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 8px;
        color: #555555;
    }
    
    /* Botones Industriales Personalizados */
    div.stButton > button {
        background-color: #FF9F1A !important;
        color: #080D14 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: bold !important;
        font-size: 14px !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 10px rgba(255, 159, 26, 0.3) !important;
    }
    
    div.stButton > button:hover {
        background-color: #ffb443 !important;
        box-shadow: 0 0 18px rgba(255, 159, 26, 0.6) !important;
        transform: translateY(-1px);
    }
    
    /* Estilización del Campo de Entrada de Chat nativo */
    .stChatInputContainer {
        border: 1px solid #3A5A70 !important;
        background-color: #080D14 !important;
        border-radius: 8px !important;
    }
    
    textarea {
        color: #C0C0C0 !important;
        background-color: transparent !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 16px !important;
    }
    
    /* Enlaces interactivos estilo HUD */
    .hud-link {
        color: #C0C0C0;
        text-decoration: none;
        font-size: 14px;
        line-height: 1.8;
    }
    
    .hud-link span {
        color: #00E5FF;
        font-weight: bold;
        text-decoration: underline;
        cursor: pointer;
    }
    
    .hud-link span:hover {
        color: #ffffff;
        text-shadow: 0 0 8px rgba(0,229,255,0.8);
    }
    
    /* Estilos del visor de Documentos */
    .doc-card {
        background-color: #151D29;
        border: 1px solid #2B2B2B;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .doc-icon {
        color: #C0C0C0;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER HUD SUPERIOR ---
now = datetime.datetime.now()
st.markdown(f"""
<div style="background-color: #05080C; border-bottom: 2px solid #3A5A70; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="color: #FF9F1A; font-size: 24px;">⚓</span>
        <span style="font-family: 'Orbitron', sans-serif; font-size: 20px; font-weight: bold; color: #C0C0C0;">
            PETRO-ASSIST <span style="font-size: 14px; color: #FF9F1A;">| Agente Operativo</span>
        </span>
    </div>
    <div style="font-family: 'Roboto Mono', monospace; font-size: 13px; color: #C0C0C0; text-align: right;">
        Bienvenido, <b>Juan Pérez (Operador)</b> | 14:32 (UTC-5) | 
        <span style="color: #00E5FF;">Consultas: {len(st.session_state.messages) // 2}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN DE DISPOSICIÓN EN 3 COLUMNAS ---
col_left, col_mid, col_right = st.columns([1, 2, 1], gap="medium")


# COLUMNA IZQUIERDA: PANEL DE CONTROL DE SESIÓN

with col_left:
    st.markdown('<div class="petro-card">', unsafe_allow_html=True)
    
    # Botón de reinicio de la consola (Iniciar Consulta)
    reiniciar = st.button("💬 INICIAR CONSULTA")
    if reiniciar:
        st.session_state.messages = [
            {"role": "assistant", "content": "Bienvenido Juan, ¿en qué puedo ayudarte hoy?"}
        ]
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sección de Gobernanza de Roles
    st.markdown('<div class="petro-card">', unsafe_allow_html=True)
    st.markdown("<h4>ACCESO Y GOBERNANZA</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; color:#555555; margin-bottom:15px;'>Estado de credenciales activas:</p>", unsafe_allow_html=True)
    
    # Tarjeta 1: Operador (Activo)
    st.markdown("""
    <div class="role-card-active">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px;">👷</span>
            <div>
                <b style="color: #00E5FF;">OPERADOR</b><br>
                <span style="font-size: 11px; color: #C0C0C0;">Acceso limitado</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tarjeta 2: Supervisor (Desactivado)
    st.markdown("""
    <div class="role-card-inactive">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px; filter: grayscale(100%);">💼</span>
            <div>
                <b>SUPERVISOR</b><br>
                <span style="font-size: 11px;">Acceso Total</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tarjeta 3: Ingeniero (Desactivado)
    st.markdown("""
    <div class="role-card-inactive">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px; filter: grayscale(100%);">🔬</span>
            <div>
                <b>INGENIERO</b><br>
                <span style="font-size: 11px;">Acceso y delete</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Resumen del Nivel de Acceso Técnico
    st.markdown('<div class="petro-card" style="background: #15202E; border-color: #00E5FF;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <span style="font-size: 45px; color: #00E5FF;">🛡️</span>
        <h5 style="color: #C0C0C0; margin-top: 10px; font-family: 'Orbitron', sans-serif;">NIVEL DE ACCESO: OPERADOR</h5>
        <p style="font-size: 13px; color: #C0C0C0; line-height: 1.4; margin-top: 5px;">
            Acceso limitado a consultas y base de conocimiento activa. No cuenta con permisos para subir, editar o eliminar registros de la red.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



# COLUMNA CENTRAL: INTERFAZ DE CHAT ASISTENTE DE IA

with col_mid:
    # Contenedor principal de la pantalla de chat con fondo tecnológico
    st.markdown("""
    <div style="background: linear-gradient(180deg, #0F1621 0%, #080D14 100%); border: 1px solid #3A5A70; border-radius: 4px; padding: 20px; min-height: 550px; box-shadow: inset 0 0 15px rgba(0,0,0,0.6);">
        <h4 style="border-bottom: 1px solid #3A5A70; padding-bottom: 10px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
            <span>🤖</span> MONITOR ASISTENTE PETRO-ASSIST
        </h4>
    """, unsafe_allow_html=True)
    
    # Renderizar el historial de chat con las plantillas de alta fidelidad visual
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            # Burbuja estilo Petro-Assist (Cian)
            st.markdown(f"""
            <div style="display: flex; gap: 15px; margin-bottom: 20px; align-items: flex-start;">
                <div style="background: linear-gradient(135deg, #0a2533, #103b50); border: 2px solid #00E5FF; border-radius: 50%; width: 45px; height: 45px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 10px rgba(0,229,255,0.4); flex-shrink: 0;">
                    <span style="font-size: 20px;">🤖</span>
                </div>
                <div>
                    <div style="font-family: 'Orbitron', sans-serif; font-size: 12px; color: #00E5FF; font-weight: bold; margin-bottom: 4px;">Petro-Assist</div>
                    <div style="background: #0A2533; border: 1px solid #00E5FF; border-radius: 4px; padding: 12px 16px; color: #C0C0C0; font-size: 14px; max-width: 90%; line-height: 1.5; box-shadow: 0 0 8px rgba(0,229,255,0.15);">
                        {message["content"]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Burbuja estilo Juan Pérez (Azul Marino / Alineado a la derecha)
            st.markdown(f"""
            <div style="display: flex; gap: 15px; margin-bottom: 20px; align-items: flex-start; justify-content: flex-end;">
                <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end;">
                    <div style="font-family: 'Orbitron', sans-serif; font-size: 12px; color: #FF9F1A; font-weight: bold; margin-bottom: 4px;">Juan Pérez</div>
                    <div style="background: #102A44; border: 1px solid #3A5A70; border-radius: 4px; padding: 12px 16px; color: #C0C0C0; font-size: 14px; text-align: left; max-width: 90%; line-height: 1.5;">
                        {message["content"]}
                    </div>
                </div>
                <div style="background: linear-gradient(135deg, #102a44, #1b3d61); border: 2px solid #3A5A70; border-radius: 50%; width: 45px; height: 45px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <span style="font-size: 20px;">👷</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)

# Input de chat nativo en la parte inferior de la columna de chat
prompt = st.chat_input("Escribe tu consulta operativa aquí...")



# COLUMNA DERECHA: BASE DE CONOCIMIENTO (FAQ & REFERENCIAS)

with col_right:
    # Sección 1: Preguntas Frecuentes
    st.markdown('<div class="petro-card">', unsafe_allow_html=True)
    st.markdown("<h4>PREGUNTAS FRECUENTES</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; color:#555555; margin-bottom:15px;'>Consultas rápidas estructuradas:</p>", unsafe_allow_html=True)
    
    # Enlaces interactivos con estilo HUD
    st.markdown('<p class="hud-link">1. ¿Cómo calibrar el sensor PT-205? <span id="link-1">Link</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="hud-link">2. Procedimiento de parada de emergencia Alpha <span id="link-2">Link</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="hud-link">3. Mantenimiento programado Turbina T-400 <span id="link-3">Links</span></p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sección 2: Documentos de Referencia
    st.markdown('<div class="petro-card">', unsafe_allow_html=True)
    st.markdown("<h4>DOCUMENTOS DE REFERENCIA</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; color:#555555; margin-bottom:15px;'>Base de datos sincronizada localmente:</p>", unsafe_allow_html=True)
    
    # Escanear y listar dinámicamente los documentos del repositorio
    documentos = obtener_documentos_activos("data")
    if documentos:
        for doc in documentos:
            st.markdown(f"""
            <div class="doc-card">
                <span class="doc-icon">📄</span>
                <div>
                    <div style="font-size: 13px; font-weight: bold; color: #C0C0C0; word-break: break-all;">{doc}</div>
                    <div style="font-size: 10px; color: #555555;">Sincronizado Local</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Sin documentos en 'data'")
        
    st.markdown('</div>', unsafe_allow_html=True)


# --- PROCESAMIENTO ACTIVO DEL CEREBRO CON MEMORIA ---
if prompt:
    # 1. Registrar y mostrar mensaje de usuario en el historial interno
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Recargar la interfaz para dibujar inmediatamente el mensaje del usuario
    st.rerun()

# Si el último mensaje es de un usuario, procesamos la respuesta de la IA
if st.session_state.messages[-1]["role"] == "user":
    pregunta_activa = st.session_state.messages[-1]["content"]
    
    try:
        # Reconstruir historial compatible en formato LangChain
        historial_langchain = []
        for msg in st.session_state.messages[:-1]: # Excluimos la pregunta actual que pasamos directo
            if msg["role"] == "user":
                historial_langchain.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                historial_langchain.append(AIMessage(content=msg["content"]))

        # Estructurar estado inicial para el orquestador del grafo
        input_state = {
            "pregunta": pregunta_activa,
            "messages": historial_langchain
        }
        
        # Invocar la lógica avanzada en la nube
        with st.spinner("Conectando con el Servidor Operativo OCI..."):
            estado_final = grafo_operativo.invoke(input_state)
            
        respuesta = estado_final.get("respuesta_RAG", "El sistema no pudo recuperar una directriz técnica segura.")
        
    except Exception as e:
        respuesta = f"❌ ERROR DE EJECUCIÓN OPERATIVA: No se pudo comunicar con el grafo del agente. Detalle: {e}"
    
    # Registrar respuesta del asistente y recargar
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    st.rerun()
