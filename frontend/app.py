import os
import re
import datetime
import requests
import streamlit as st


# Configuración de variables

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
API_KEY = os.getenv("API_KEY", "dev-key-cambiar-en-produccion")
USER_NAME = os.getenv("USER_NAME", "Juan Pérez")
USER_ROLE = os.getenv("USER_ROLE", "operador")

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "assets", "templates")
CSS_DIR = os.path.join(BASE_DIR, "assets", "css")


# Carga de Assets Externos

def load_template(name: str) -> str:
    """Lee un archivo HTML desde assets/templates/."""
    path = os.path.join(TEMPLATES_DIR, f"{name}.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"<!-- Template '{name}.html' no encontrado -->"

def render_template(name: str, **kwargs) -> str:
    """Carga un template, reemplaza {{clave}} y limpia para Streamlit."""
    html = load_template(name)
    for key, value in kwargs.items():
        html = html.replace(f"{{{{{key}}}}}", str(value))
    return html

def minify_html(html: str) -> str:
    """
    Limpia el HTML para que Streamlit no lo interprete como código Markdown.
    Elimina saltos de línea y espacios excesivos entre tags.
    """
    # Quitar comentarios HTML
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    # Reemplazar saltos de línea y espacios múltiples entre tags por un espacio
    html = re.sub(r">\\s+<", "><", html)
    # Quitar espacios al inicio/fin
    html = html.strip()
    return html

def load_css() -> str:
    """Lee styles.css desde assets/css/."""
    path = os.path.join(CSS_DIR, "styles.css")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.warning("⚠️ styles.css no encontrado.")
        return ""

# Incialización de estado de sesión

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bienvenido, soy el Agente-OCI. ¿En qué puedo ayudarte hoy?"}
    ]
if "consult_count" not in st.session_state:
    st.session_state.consult_count = 0


# CConfiguracion de la página Streamlit

st.set_page_config(
    page_title="AGENTE-OCI | Asistente Operativo",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Carga CSS externo

css_content = load_css()
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# Header: Nombre de usuario, rol y contador de consultas

now = datetime.datetime.now()
header_html = render_template(
    "header",
    user_name=USER_NAME,
    user_role=USER_ROLE.capitalize(),
    consult_count=st.session_state.consult_count,
    timestamp=now.strftime("%H:%M (UTC-5) %b %d, %Y")
)
# Usar st.html() para evitar que Markdown interprete el HTML como código
st.html(minify_html(header_html))


# Cuerpo principal: tres columnas (izquierda, central, derecha)

col_left, col_mid, col_right = st.columns([1, 2.2, 1], gap="medium")


# Sidebar-left: Panel de Control

with col_left:
    # Botón nativo de Streamlit (estilizado por CSS externo)
    if st.button("💬 INICIAR CONSULTA", key="btn_consulta", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Bienvenido, soy el Agente-OCI. ¿En qué puedo ayudarte hoy?"}
        ]
        st.session_state.consult_count = 0
        st.rerun()
    
    # Determinar clases de roles según USER_ROLE
    role_classes = {
        "operador":   ("active", "inactive", "inactive"),
        "supervisor": ("inactive", "active", "inactive"),
        "ingeniero":  ("inactive", "inactive", "active")
    }
    op_cls, sup_cls, ing_cls = role_classes.get(USER_ROLE, ("active", "inactive", "inactive"))
    
    role_colors = {
        "operador": "#e89b2a",
        "supervisor": "#38bdf8",
        "ingeniero": "#10b981"
    }
    
    descriptions = {
        "operador": "Acceso limitado a consultas y base de conocimiento activa. No permite subir, editar o eliminar registros de la red.",
        "supervisor": "Acceso total a consultas y gestión de documentos. Puede subir nuevos registros a la base de conocimiento.",
        "ingeniero": "Acceso de administrador. Puede consultar, subir, editar y eliminar registros de la red."
    }
    
    sidebar_left_html = render_template(
        "sidebar-left",
        operador_class=op_cls,
        supervisor_class=sup_cls,
        ingeniero_class=ing_cls,
        role_name=USER_ROLE.upper(),
        role_color=role_colors.get(USER_ROLE, "#e89b2a"),
        access_description=descriptions.get(USER_ROLE, "Acceso limitado.")
    )
    st.html(minify_html(sidebar_left_html))


# Chat-center: Chat del Agente-OCI

with col_mid:
    # Generar HTML de mensajes dinámicamente (en una sola línea para evitar bloques de código)
    messages_html = ""
    for msg in st.session_state.messages:
        content_escaped = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        if msg["role"] == "assistant":
            messages_html += (
                '<div class="message">'
                '<div class="message-avatar bot">'
                '<svg viewBox="0 0 24 24" fill="currentColor">'
                '<circle cx="12" cy="8" r="3"/>'
                '<rect x="8" y="4" width="8" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/>'
                '<path d="M6 14h12v3a2 2 0 01-2 2H8a2 2 0 01-2-2v-3z" fill="none" stroke="currentColor" stroke-width="1.5"/>'
                '<circle cx="9" cy="8" r="0.8" fill="currentColor"/>'
                '<circle cx="15" cy="8" r="0.8" fill="currentColor"/>'
                '</svg>'
                '</div>'
                '<div class="message-content">'
                '<div class="message-sender">Agente-OCI</div>'
                f'<div class="message-bubble bot">{content_escaped}</div>'
                '</div>'
                '</div>'
            )
        else:
            messages_html += (
                '<div class="message user">'
                '<div class="message-avatar user">'
                '<svg viewBox="0 0 24 24" fill="currentColor">'
                '<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>'
                '</svg>'
                '</div>'
                '<div class="message-content">'
                f'<div class="message-sender">{USER_NAME}</div>'
                f'<div class="message-bubble user">{content_escaped}</div>'
                '</div>'
                '</div>'
            )
    
    # Renderizar chat-center con mensajes inyectados
    chat_html = render_template("chat-center", messages=messages_html)
    st.html(minify_html(chat_html))
    
    # Input nativo de Streamlit (estilizado por CSS externo)
    prompt = st.chat_input("Escribe tu consulta operativa aquí...")


# Sidebar-right: Base de Conocimiento

with col_right:
    # FAQ estática
    faq_html = (
        '<div class="faq-item">'
        '<span class="faq-number">1.</span>'
        '<div class="faq-text">¿Cómo calibrar sensor PT-205? <a href="#">Link</a></div>'
        '</div>'
        '<div class="faq-item">'
        '<span class="faq-number">2.</span>'
        '<div class="faq-text">Procedimiento parada emergencia Alpha <a href="#">Link</a></div>'
        '</div>'
        '<div class="faq-item">'
        '<span class="faq-number">3.</span>'
        '<div class="faq-text">Mantenimiento Turbina T-400. <a href="#">Links</a></div>'
        '</div>'
    )
    
    # Documentos: intentar obtener del backend, fallback offline
    docs_html = ""
    try:
        resp = requests.get(f"{API_BASE_URL}/documents", headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            docs = resp.json()
            for doc in docs:
                date = doc.get("updated", "")[:10] or "Sincronizado"
                docs_html += (
                    '<div class="doc-item">'
                    '<div class="doc-icon">'
                    '<svg viewBox="0 0 24 24" fill="currentColor">'
                    '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z"/>'
                    '<path d="M14 3v5h5M16 13H8M16 17H8M10 9H8" stroke="currentColor" stroke-width="1.5" fill="none"/>'
                    '</svg>'
                    '</div>'
                    '<div class="doc-info">'
                    f'<div class="doc-title">{doc["name"]}</div>'
                    f'<div class="doc-date">{date}</div>'
                    '</div>'
                    '</div>'
                )
        else:
            docs_html = '<div style="color:#5a6b7d;font-size:11px;padding:8px 0;">Backend no responde.</div>'
    except Exception:
        docs_html = '<div style="color:#5a6b7d;font-size:11px;padding:8px 0;">Sin conexión al backend.</div>'
    
    sidebar_right_html = render_template(
        "sidebar-right",
        faq_items=faq_html,
        doc_items=docs_html
    )
    st.html(minify_html(sidebar_right_html))
    
    # Panel de gestión de documentos (solo supervisor/ingeniero)
    if USER_ROLE in ("supervisor", "ingeniero"):
        st.markdown('<div class="divider" style="margin:12px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-bottom:10px;">Gestión de Documentos</div>', unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Subir documento",
            type=["pdf", "xlsx", "xls", "docx", "txt"],
            label_visibility="collapsed"
        )
        if uploaded is not None:
            with st.spinner("Subiendo a GCS..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    upload_resp = requests.post(
                        f"{API_BASE_URL}/documents/upload",
                        headers={"X-API-Key": API_KEY},
                        files=files,
                        timeout=60
                    )
                    if upload_resp.status_code == 200:
                        st.success(f"✅ {uploaded.name} subido correctamente.")
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {upload_resp.text}")
                except Exception as e:
                    st.error(f"❌ Error de conexión: {e}")
    
    # Botón reindexar (solo ingeniero)
    if USER_ROLE == "ingeniero":
        if st.button("🔄 Reindexar Vectores", use_container_width=True):
            with st.spinner("Reindexando vectores..."):
                try:
                    reindex_resp = requests.post(
                        f"{API_BASE_URL}/documents/reindex",
                        headers=HEADERS,
                        timeout=120
                    )
                    if reindex_resp.status_code == 200:
                        st.success("✅ Vectores reindexados exitosamente.")
                    else:
                        st.error(f"❌ Error: {reindex_resp.text}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")


# Input de usuario y envío al backend

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    pregunta_activa = st.session_state.messages[-1]["content"]
    
    with st.spinner("Conectando con Servidor Operativo OCI..."):
        try:
            payload = {
                "question": pregunta_activa,
                "history": st.session_state.messages[:-1],
                "user_id": USER_NAME,
                "role": USER_ROLE
            }
            response = requests.post(
                f"{API_BASE_URL}/chat",
                headers=HEADERS,
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                respuesta = data.get("answer", "El sistema no pudo recuperar una directriz técnica segura.")
                st.session_state.consult_count += 1
            else:
                respuesta = f"❌ Error del servidor ({response.status_code}): {response.text[:200]}"
        except requests.exceptions.ConnectionError:
            respuesta = "❌ ERROR DE CONEXIÓN: No se pudo contactar el backend. Verifique que el servicio esté activo."
        except requests.exceptions.Timeout:
            respuesta = "⏱️ TIMEOUT: El servidor tardó demasiado en responder. Intente de nuevo."
        except Exception as e:
            respuesta = f"❌ ERROR DE EJECUCIÓN: {str(e)}"
    
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
    st.rerun()
