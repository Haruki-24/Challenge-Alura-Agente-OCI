# --- 1. CONFIGURACIÓN E IMPORTACIONES ---

import os
from pathlib import Path
from typing import TypedDict, Optional, Dict, Literal, List
from pydantic import BaseModel, Field

# Componentes de LangChain para procesamiento, cargadores y vectores
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Reranking local y HuggingFace
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

# Orquestador de agentes LangGraph
from langgraph.graph import START, END, StateGraph

# Carga de variables de entorno locales (.env)
from dotenv import load_dotenv
load_dotenv()

# Validación de la clave API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError(
        "¡Error! No se encontró la variable GEMINI_API_KEY en tu archivo '.env'.\n"
        "Por favor, asegúrate de crear el archivo '.env' en la raíz con tus credenciales."
    )

# Configuración de proveedores según el diseño del desafío
LLM_PROVIDER = "google"
LLM_MODEL_NAME = "gemini-3.1-flash-lite" 

EMBEDDING_PROVIDER = "huggingface"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Inicialización dinámica del modelo LLM
print(f"Configurando LLM: {LLM_PROVIDER.upper()} ({LLM_MODEL_NAME})...")

if LLM_PROVIDER == "google":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Falta GEMINI_API_KEY en el archivo .env")
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME, 
        temperature=0.1, 
        google_api_key=api_key
    )
elif LLM_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Falta OPENAI_API_KEY en el archivo .env")
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME, 
        temperature=0.1, 
        openai_api_key=api_key
    )
elif LLM_PROVIDER == "cohere":
    from langchain_cohere import ChatCohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Falta COHERE_API_KEY en el archivo .env")
    llm = ChatCohere(
        model=LLM_MODEL_NAME, 
        temperature=0.1, 
        cohere_api_key=api_key
    )
else:
    raise ValueError(f"Proveedor de LLM no soportado: {LLM_PROVIDER}")


# --- 2. PROCESADO DE DATOS Y VECTORIZACIÓN ----

RUTA_DATOS = "./data/"
documentos_oci = []

if os.path.exists(RUTA_DATOS):
    print(" Cargando documentos del almacén de datos local...")
    loader_pdfs = DirectoryLoader(RUTA_DATOS, glob="*.pdf", loader_cls=PyPDFLoader)
    loader_excel = DirectoryLoader(RUTA_DATOS, glob="*.xlsx", loader_cls=UnstructuredExcelLoader)
    
    all_loaders = MergedDataLoader(loaders=[loader_pdfs, loader_excel])
    try:
        documentos_oci = all_loaders.load()
        print(f" Carga completada. Total de documentos en crudo: {len(documentos_oci)}")
    except Exception as e:
        print(f" ¡Error! cargando el set de datos: {e}")
else:
    print(f" *Alerta*: Carpeta '{RUTA_DATOS}' no encontrada. Créala en la raíz y coloca tus manuales.")

retriever = None
text_splitter = None
REQUIERE_PREFIJO_E5 = (EMBEDDING_PROVIDER == "huggingface" and "e5" in EMBEDDING_MODEL_NAME.lower())

if documentos_oci:
    print(f" Configurando segmentación adaptada a {EMBEDDING_PROVIDER.upper()}...")
    
    if EMBEDDING_PROVIDER == "huggingface":
        print(f" Descargando tokenizador local para '{EMBEDDING_MODEL_NAME}'...")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=400,
            chunk_overlap=50
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

    documentos_fragmentados = text_splitter.split_documents(documentos_oci)

    if REQUIERE_PREFIJO_E5:
        print(" Modelo E5 detectado. Aplicando prefijo 'passage: ' para indexación eficiente...")
        for doc in documentos_fragmentados:
            if not doc.page_content.startswith("passage: "):
                doc.page_content = f"passage: {doc.page_content}"

    print(" Generando representaciones vectoriales (Embeddings)...")
    if EMBEDDING_PROVIDER == "huggingface":
        embeddings_local = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        embeddings_local = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    print(" Indexando base de datos vectorial local FAISS...")
    FAISS_DB_PATH = "./faiss_index_oci"
    vectorstore = FAISS.from_documents(documentos_fragmentados, embeddings_local)
    vectorstore.save_local(FAISS_DB_PATH)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    print(" Indexación completada y persistida de forma local.")

print(" Inicializando modelo de Reranking de alta precisión (MiniLM-L-6-v2)...")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


#  --- 3. LÓGICA DE TRIAJE OPERATIVO (CLASIFICACIÓN CON VALIDACIÓN) ---

class TriajeOperativoOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ALERTAR_SUPERVISOR"] = Field(
        description="Dirección lógica sugerida para procesar el caso en el flujo."
    )
    urgency: Literal["BAJA", "MEDIANA", "ALTA"] = Field(
        description="Criticidad del evento o tarea reportada en campo."
    )
    motivo: str = Field(
        description="Justificación detallada detrás de la categorización."
    )

PROMPT_TRIAJE_OPERATIVO = """
Eres el Agente Inteligente de Triaje Operativo y de Riesgos en Campo (OCI).
Tu tarea es analizar el mensaje del usuario y clasificarlo técnicamente de forma precisa.

CAPACIDADES TÉCNICAS Tú tienes acceso completo a la base de datos documental RAG de la empresa, la cual contiene:
- Cronograma de Intervenciones de Pulling de todos los pozos (con fechas estimadas, duraciones, cuadrillas).
- Programa de Mantenimiento Preventivo de todos los pozos y componentes.
- Inventario completo de recursos, EPP, herramientas y materiales.
- Manuales, políticas HSE y procedimientos del supervisor.

Reglas de Decisión (`decision`):
- "AUTO_RESOLVER": Debe seleccionarse para consultas informativas, procedimientos de bajo riesgo, manuales, stock, listados generales de planificación,
  o cronogramas completos (ej: "¿qué pozos tienen tareas de mantenimiento programadas?", "cronograma de pulling", "cuándo intervienen el pozo P-404").
  Si el usuario pide un listado, resumen o fecha que esté en la planificación cargada, la decisión SIEMPRE debe ser "AUTO_RESOLVER".
- "PEDIR_INFO": Debe usarse ÚNICAMENTE si la pregunta del usuario es extremadamente ambigua y carece de cualquier punto de referencia para buscar en el RAG
  (ej: "¿está lista mi herramienta?" sin decir qué herramienta, o "¿cuál es el estado?" sin especificar de qué pozo, tarea o equipo se habla).
  NO uses esta categoría para preguntas de listado general (como "¿qué pozos tienen mantenimiento?"),
  ya que esa información consolidada se encuentra directamente en los documentos de la base de datos RAG.
- "ALERTAR_SUPERVISOR": Situaciones críticas de seguridad (HSE), alertas por tormentas o viento severo, accidents,
  operaciones con tensión eléctrica, o decisiones que involucren un riesgo físico inminente en el campo.

Reglas de Urgencia (`urgency`):
- "BAJA": Consultas generales sin impacto directo en la producción inmediata.
- "MEDIANA": Planificación de mantenimiento preventivo, reposición de herramientas menores o control de stock rutinario.
- "ALTA": Emergencias de seguridad (HSE), paradas imprevistas de pozos, tormentas eléctricas en área operativa, o falta de EPP crítico para una maniobra inmediata.

Devuelve SOLO el objeto JSON estructurado según el esquema solicitado.
"""

chain_de_triaje_operativo = llm.with_structured_output(TriajeOperativoOut)

def ejecutar_triaje_operativo(mensaje: str) -> Dict:
    salida: TriajeOperativoOut = chain_de_triaje_operativo.invoke([
        SystemMessage(content=PROMPT_TRIAJE_OPERATIVO),
        HumanMessage(content=mensaje)
    ])
    return salida.model_dump()


# --- 4. COMPONENTES RAG AVANZADO (RERANKING & EVALUACIÓN DE ALUCINACIONES) ---

def ejecutar_reranking(query: str, documentos: list, top_n: int = 3) -> list:
    if not documentos:
        return []
    pares = [[query, doc.page_content.replace("passage: ", "").strip()] for doc in documentos]
    scores = reranker_model.predict(pares)
    
    documentos_con_score = list(zip(documentos, scores))
    documentos_ordenados = sorted(documentos_con_score, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in documentos_ordenados[:top_n]]

def ensamblar_contexto(documentos_reranked: list) -> tuple:
    if not documentos_reranked:
        return "No hay información contextual disponible.", []
    
    contexto_bloque = ""
    citaciones_limpias = []
    for idx, doc in enumerate(documentos_reranked, 1):
        origen = doc.metadata.get('source', 'Desconocido').split('/')[-1] if 'source' in doc.metadata else doc.metadata.get('file_path', 'Documento')
        pagina = doc.metadata.get('page', None)
        
        citacion_label = f"{origen} (Pág. {pagina + 1})" if pagina is not None else origen
        citaciones_limpias.append(citacion_label)
        
        contenido = doc.page_content.replace("passage: ", "").strip()
        contexto_bloque += f"--- [RECURSO {idx} | Fuente: {citacion_label}] ---\n{contenido}\n\n"
    
    return contexto_bloque, citaciones_limpias

prompt_rag_operativo = ChatPromptTemplate([
    ("system",
     """
     Eres el Ingeniero de Soporte Operativo y Mentor de Seguridad (HSE) de la empresa OCI.
     Tu rol es responder la consulta utilizando únicamente el contexto documental provisto.

     REGLAS DE RESPUESTA:
     1. Utiliza un tono profesional, claro y orientado a la seguridad industrial.
     2. Si la consulta involucra procedimientos críticos de mantenimiento o campo, añade siempre un apartado de: "⚠️ ADVERTENCIA DE SEGURIDAD (HSE)".
     3. Si en el contexto técnico proporcionado NO figura la información para responder, contesta estrictamente: "No cuento con registros técnicos suficientes en mis manuales para resolver esta consulta de forma segura."
     4. Finaliza tus recomendaciones técnicas recordando al usuario que el Supervisor de Campo tiene la decisión final frente a cualquier imprevisto técnico o climático.
     """),
    ("human", "Contexto técnico recuperado:\n{context}\n\nPregunta del operador en campo: {input}")
])

document_chain_operativa = create_stuff_documents_chain(llm, prompt=prompt_rag_operativo)


# --- GUARDRAIL DE ALUCINACIONES ---
class EvaluacionFidelidad(BaseModel):
    fiel_al_contexto: bool = Field(
        description="True si la respuesta propuesta se basa exclusivamente en los documentos de soporte proporcionados."
    )
    justificacion: str = Field(
        description="Explicación detallada de la auditoría."
    )

evaluador_alucinaciones = llm.with_structured_output(EvaluacionFidelidad)

def validar_respuesta_frente_al_contexto(pregunta: str, contexto: str, respuesta: str) -> EvaluacionFidelidad:
    prompt_evaluacion = f"""
    Analiza rigurosamente si la 'Respuesta Propuesta' incurre en alucinaciones basadas en la información provista en el 'Contexto Técnico'.

    EXCEPCIÓN DE SEGURIDAD CORPORATIVA (CRÍTICO):
    Los recordatorios obligatorios de seguridad (HSE) y advertencias de que "el Supervisor de Campo tiene la decisión final" son políticas operativas generales requeridas en todas las respuestas de OCI. NO debes considerarlos como una alucinación bajo ningún motivo, incluso si esas palabras exactas no se mencionan en el 'Contexto Técnico'.
    Solo debes calificar la respuesta como infiel (False) si inventa hechos técnicos concretos (como fechas de pulling falsas, números de pozos inexistentes en el contexto, stock de herramientas inventado, etc.).

    [Contexto Técnico]
    {contexto}

    [Pregunta del Operador]
    {pregunta}

    [Respuesta Propuesta]
    {respuesta}

    Devuelve la evaluación estructurada según el esquema solicitado.
    """
    return evaluador_alucinaciones.invoke(prompt_evaluacion)

def resolver_con_rag_avanzado(pregunta: str) -> Dict:
    if not retriever:
        return {
            "respuesta": "No cuento con registros técnicos suficientes en mis manuales para resolver esta consulta de forma segura.",
            "citaciones": [],
            "rag_exito": False,
            "motivo_fallo": "Búsqueda vectorial no inicializada por falta de documentos."
        }
    
    candidatos = retriever.invoke(pregunta) if not REQUIERE_PREFIJO_E5 else retriever.invoke(f"query: {pregunta}")
    documentos_filtrados = ejecutar_reranking(pregunta, candidatos, top_n=3)
    
    if not documentos_filtrados:
        return {
            "respuesta": "No cuento con registros técnicos suficientes en mis manuales para resolver esta consulta de forma segura.",
            "citaciones": [],
            "rag_exito": False,
            "motivo_fallo": "No se encontraron documentos candidatos relevantes."
        }
    
    contexto_final, citaciones = ensamblar_contexto(documentos_filtrados)
    respuesta_ia = document_chain_operativa.invoke({
        "input": pregunta,
        "context": documentos_filtrados
    })
    
    if "no cuento con registros técnicos" in respuesta_ia.lower():
        return {
            "respuesta": respuesta_ia,
            "citaciones": [],
            "rag_exito": False,
            "motivo_fallo": "La base de conocimientos no cubre el procedimiento solicitado."
        }
    
    evaluacion = validar_respuesta_frente_al_contexto(pregunta, contexto_final, respuesta_ia)
    if not evaluacion.fiel_al_contexto:
        print(f" [Filtro de Alucinación] Bloqueada respuesta infiel: {evaluacion.justificacion}")
        return {
            "respuesta": "No cuento con registros técnicos suficientes en mis manuales para resolver esta consulta de forma segura.",
            "citaciones": [],
            "rag_exito": False,
            "motivo_fallo": f"Alucinación detectada: {evaluacion.justificacion}"
        }
    
    return {
        "respuesta": respuesta_ia,
        "citaciones": citaciones,
        "rag_exito": True
    }


# --- 5. ORQUESTACIÓN DE AGENTE CON LANGGRAPH Y MEMORIA CONVERSACIONAL ---

class AgentState(TypedDict):
    """Bus de memoria compartida con soporte nativo para historial de chat."""
    pregunta: str                      # La última pregunta cruda del usuario
    messages: List[BaseMessage]        # Historial completo de la sesión de chat
    pregunta_condensada: str           # Pregunta reescrita contextualizada históricamente
    triaje: Dict
    respuesta_RAG: Optional[str]
    citaciones: Optional[List]
    rag_exito: bool
    accion_final: str


# --- DEFINICION DE NODOS DEL GRAFO ---

def nodo_condensar_pregunta(state: AgentState) -> Dict:
    """Revisa el historial acumulado en el estado y unifica el contexto."""
    historial = state.get("messages", [])
    ultima_pregunta = state["pregunta"]
    
    # Si es el primer mensaje del chat, no hay nada que condensar
    if len(historial) <= 1:
        return {"pregunta_condensada": ultima_pregunta}
    
    prompt_memoria = ChatPromptTemplate.from_messages([
        ("system", (
            "Dada la siguiente conversación histórica en un entorno industrial (OCI) y una nueva pregunta de seguimiento, "
            "reescríbela para que sea una pregunta independiente y clara. Debe contener de forma explícita nombres de pozos, "
            "equipos, herramientas o incidentes de seguridad mencionados anteriormente. NO respondas la consulta, solo redáctala de nuevo de forma técnica."
        )),
        ("placeholder", "{chat_history}"),
        ("human", "Pregunta de seguimiento: {input}")
    ])
    
    cadena_memoria = prompt_memoria | llm
    pregunta_rica = cadena_memoria.invoke({
        "chat_history": historial[:-1], # Excluimos el último mensaje para evitar redundancia
        "input": ultima_pregunta
    })
    
    return {"pregunta_condensada": pregunta_rica.content}


def nodo_triaje_operaciones(state: AgentState) -> AgentState:
    print("[NODO: Triaje] Analizando riesgos de la consulta...")
    # Realizamos el triaje sobre la pregunta condensada para capturar el contexto real histórico
    resultado_triaje = ejecutar_triaje_operativo(state["pregunta_condensada"])
    return {"triaje": resultado_triaje}


def nodo_auto_resolver_operaciones(state: AgentState) -> AgentState:
    print("[NODO: Auto-Resolver] Consultando manuales técnicos...")
    # El RAG, Reranker y Guardrails operan sobre la pregunta condensada con memoria
    resultado_rag = resolver_con_rag_avanzado(state["pregunta_condensada"])
    return {
        "respuesta_RAG": resultado_rag["respuesta"],
        "citaciones": resultado_rag["citaciones"],
        "rag_exito": resultado_rag["rag_exito"],
        "accion_final": "RESOLUCIÓN_AUTOMÁTICA" if resultado_rag["rag_exito"] else "ESCALADO"
    }


def nodo_solicitar_detalles(state: AgentState) -> AgentState:
    print("[NODO: Solicitar Detalles] Formulando plantilla de requerimientos...")
    motivo = state["triaje"].get("motivo", "Falta de información clave.")
    respuesta = (
        f"**ASISTENTE OCI**: Necesitamos datos adicionales para poder asesorarte.\n"
        f"**Razón**: {motivo}\n\n"
        f"Por favor, facilítanos la siguiente información si aplica:\n"
        f"- ID o Nombre del Pozo (ej. Pozo PCP-04)\n"
        f"- Tipo de mantenimiento / Herramienta involucrada\n"
        f"- ¿Se encuentra en zona de riesgo o bajo condiciones climáticas adversas?"
    )
    return {
        "respuesta_RAG": respuesta,
        "citaciones": [],
        "rag_exito": False,
        "accion_final": "SOLICITAR_DETALLES"
    }


def nodo_alertar_supervisor(state: AgentState) -> AgentState:
    print("[NODO: Alerta Supervisor] Ejecutando protocolo de seguridad HITL (Human-in-the-Loop)...")
    resultado_rag = resolver_con_rag_avanzado(state["pregunta_condensada"])
    propuesta_datos = resultado_rag["respuesta"] if resultado_rag["rag_exito"] else "No se encontraron manuales específicos para este escenario crítico o la propuesta no superó los controles de seguridad de alucinación."

    alerta_hitl = (
        f"**ALERTA DE SEGURIDAD / ESCENARIO CRÍTICO DETECTADO** \n"
        f"**Urgencia**: {state['triaje'].get('urgency', 'ALTA')} "
        f" **Motivo**: {state['triaje'].get('motivo', 'Riesgo Técnico')}\n"
        f"----------------------------------------------------------------------\n"
        f"**Recomendación Asistida por IA (Basada en datos)**:\n"
        f"{propuesta_datos}\n"
        f"----------------------------------------------------------------------\n"
        f"**PROTOCOLO HUMAN-IN-THE-LOOP (HITL)**:\n"
        f"Esta consulta involucra operaciones con riesgos potenciales o decisiones de alta prioridad.\n"
        f"**Supervisor asignado**: Por favor evalúe el estado climático actual, el equipo de protección (EPP) y valide físicamente el procedimiento antes de autorizar la maniobra en campo.\n\n"
        f"**[ ] Aprobar Operación   |   [ ] Reconducir Protocolo   |   [ ] Solicitar Verificación de Campo**"
    )

    return {
        "respuesta_RAG": alerta_hitl,
        "citaciones": resultado_rag["citaciones"],
        "rag_exito": resultado_rag["rag_exito"],
        "accion_final": "ALERTA_SUPERVISOR_HITL"
    }


# --- DEFINICION DE ARISTAS CONDICIONALES ---

def arista_decision_triaje(state: AgentState) -> str:
    decision = state["triaje"]["decision"]
    if decision == "AUTO_RESOLVER":
        return "ir_a_rag"
    elif decision == "PEDIR_INFO":
        return "ir_a_detalles"
    elif decision == "ALERTAR_SUPERVISOR":
        return "ir_a_alerta"
    raise ValueError(f"Decisión inválida: {decision}")


def arista_evaluacion_rag(state: AgentState) -> str:
    if state["rag_exito"]:
        return "finalizar"
    
    palabras_criticas = ["riesgo", "accidente", "emergencia", "pulling", "tensión", "daño", "parada", "tormenta", "viento"]
    if any(p in state["pregunta_condensada"].lower() for p in palabras_criticas):
        print(" -> Fallo de recuperación o guardrail en tema sensible. Escalando a Supervisor.")
        return "ir_a_alerta"
    return "ir_a_detalles"


# --- CONSTRUCCIÓN DEL GRAFO ---

workflow_operativo = StateGraph(AgentState)

# Registrar nodos (añadiendo la condensación al inicio)
workflow_operativo.add_node("condensar_pregunta", nodo_condensar_pregunta)
workflow_operativo.add_node("triaje", nodo_triaje_operaciones)
workflow_operativo.add_node("auto_resolver", nodo_auto_resolver_operaciones)
workflow_operativo.add_node("pedir_info", nodo_solicitar_detalles)
workflow_operativo.add_node("alertar_supervisor", nodo_alertar_supervisor)

# Definir punto de entrada hacia el nodo de memoria
workflow_operativo.add_edge(START, "condensar_pregunta")

# Enlace de la memoria hacia el triaje estratégico
workflow_operativo.add_edge("condensar_pregunta", "triaje")

# Flujos condicionales desde el triaje
workflow_operativo.add_conditional_edges(
    "triaje",
    arista_decision_triaje,
    {
        "ir_a_rag": "auto_resolver",
        "ir_a_detalles": "pedir_info",
        "ir_a_alerta": "alertar_supervisor"
    }
)

# Flujos condicionales desde el RAG
workflow_operativo.add_conditional_edges(
    "auto_resolver",
    arista_evaluacion_rag,
    {
        "finalizar": END,
        "ir_a_alerta": "alertar_supervisor",
        "ir_a_detalles": "pedir_info"
    }
)

# Transiciones de cierre directo
workflow_operativo.add_edge("pedir_info", END)
workflow_operativo.add_edge("alertar_supervisor", END)

# Compilación final del motor del Agente
grafo_operativo = workflow_operativo.compile()


# --- BUCLE DE PRUEBA LOCAL CON HISTORIAL ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Probador Local del Cerebro con Memoria - OCI Core")
    print("Escribe 'salir' para finalizar.")
    print("="*60 + "\n")
    
    historial_memoria = []
    
    while True:
        user_input = input("Operador en Campo ---> ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            break
            
        if not user_input.strip():
            continue
            
        # 1. Agregamos el mensaje del operario a la lista histórica
        historial_memoria.append(HumanMessage(content=user_input))
        
        # 2. Invocamos el grafo pasándole tanto la pregunta actual como la lista
        inputs = {
            "pregunta": user_input,
            "messages": historial_memoria
        }
        
        resultado = grafo_operativo.invoke(inputs)
        
        print("\n⚡ [REGISTRO DE FLUJO INTERNO]")
        print(f"-> Consulta Condensada: '{resultado['pregunta_condensada']}'")
        print(f"-> Triaje: {resultado['triaje']['decision']} | Urgencia: {resultado['triaje']['urgency']}")
        print(f"-> Motivo: {resultado['triaje']['motivo']}")
        print(f"-> Citaciones: {resultado.get('citaciones', [])}\n")
        
        print(resultado["respuesta_RAG"])
        print("\n" + "="*60 + "\n")
        
        # 3. Guardamos la respuesta final de la IA en la memoria para el contexto del siguiente turno
        historial_memoria.append(AIMessage(content=resultado["respuesta_RAG"]))
