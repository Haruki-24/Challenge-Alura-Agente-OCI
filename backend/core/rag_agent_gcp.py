# --- 1. CONFIGURACIÓN E IMPORTACIONES ---
import os
import logging
from pathlib import Path
from typing import TypedDict, Optional, Dict, Literal, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Componentes de LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# LangGraph
from langgraph.graph import START, END, StateGraph

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTES GLOBALES (pueden moverse a un config) ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Falta GEMINI_API_KEY en el archivo .env")

LLM_MODEL_NAME = "gemini-1.5-flash"  # o el que tengas disponible
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
FAISS_DB_PATH = "./faiss_index_oci"
REQUIERE_PREFIJO_E5 = "e5" in EMBEDDING_MODEL_NAME.lower()

# --- DEFINICIONES DE TIPOS Y MODELOS ---
class AgentState(TypedDict):
    pregunta: str
    messages: List[BaseMessage]
    pregunta_condensada: str
    triaje: Dict
    respuesta_RAG: Optional[str]
    citaciones: Optional[List]
    rag_exito: bool
    accion_final: str

class TriajeOperativoOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ALERTAR_SUPERVISOR"]
    urgency: Literal["BAJA", "MEDIANA", "ALTA"]
    motivo: str

class EvaluacionFidelidad(BaseModel):
    fiel_al_contexto: bool
    justificacion: str

# --- PROMPTS ---
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

PROMPT_RAG_OPERATIVO = ChatPromptTemplate([
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

# --- CLASE PRINCIPAL ---
class AgenteOperativoOCI:
    def __init__(self):
        """Inicializa el agente: modelos, índice FAISS, cadenas y grafo."""
        self._inicializar_modelos()
        self._cargar_indice_faiss()
        self._inicializar_cadenas()
        self._compilar_grafo()
        logger.info("Agente OCI inicializado y listo.")

    # ---------- INICIALIZACIÓN ----------
    def _inicializar_modelos(self):
        """Carga el LLM, embeddings y reranker."""
        logger.info(f"Cargando LLM: {LLM_MODEL_NAME}")
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=0.1,
            google_api_key=GEMINI_API_KEY
        )

        logger.info(f"Cargando Embeddings: {EMBEDDING_MODEL_NAME}")
        self.embeddings_local = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("Cargando modelo de Reranking (CrossEncoder)...")
        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    def _cargar_indice_faiss(self):
        """Carga el índice FAISS desde disco."""
        if not os.path.exists(FAISS_DB_PATH):
            raise FileNotFoundError(
                f"No se encontró el índice FAISS en {FAISS_DB_PATH}. "
                "Ejecute el script de indexación primero."
            )
        logger.info("Cargando índice vectorial local FAISS...")
        self.vectorstore = FAISS.load_local(
            FAISS_DB_PATH,
            self.embeddings_local,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
        logger.info("Índice FAISS cargado.")

    def _inicializar_cadenas(self):
        """Crea las cadenas con salida estructurada y la cadena RAG."""
        self.chain_triaje = self.llm.with_structured_output(TriajeOperativoOut)
        self.chain_evaluador = self.llm.with_structured_output(EvaluacionFidelidad)
        # La cadena RAG se crea con la plantilla global
        self.document_chain = create_stuff_documents_chain(self.llm, PROMPT_RAG_OPERATIVO)

    # ---------- MÉTODOS AUXILIARES ----------
    def _ejecutar_reranking(self, query: str, documentos: list, top_n: int = 3) -> list:
        """Reordena los documentos mediante CrossEncoder."""
        if not documentos:
            return []
        pares = [
            [query, doc.page_content.replace("passage: ", "").strip()]
            for doc in documentos
        ]
        scores = self.reranker_model.predict(pares)
        documentos_con_score = list(zip(documentos, scores))
        documentos_ordenados = sorted(documentos_con_score, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in documentos_ordenados[:top_n]]

    def _ensamblar_contexto(self, documentos_reranked: list) -> tuple:
        """Convierte los documentos en un bloque de texto y lista de citas."""
        if not documentos_reranked:
            return "No hay información contextual disponible.", []

        contexto_bloque = ""
        citaciones_limpias = []
        for idx, doc in enumerate(documentos_reranked, 1):
            origen = doc.metadata.get('source', 'Desconocido').split('/')[-1]
            pagina = doc.metadata.get('page')
            citacion_label = f"{origen} (Pág. {pagina+1})" if pagina is not None else origen
            citaciones_limpias.append(citacion_label)
            contenido = doc.page_content.replace("passage: ", "").strip()
            contexto_bloque += f"--- [RECURSO {idx} | Fuente: {citacion_label}] ---\n{contenido}\n\n"
        return contexto_bloque, citaciones_limpias

    def _validar_respuesta(self, pregunta: str, contexto: str, respuesta: str) -> EvaluacionFidelidad:
        """Evalúa si la respuesta es fiel al contexto."""
        prompt_eval = f"""
        Analiza rigurosamente si la 'Respuesta Propuesta' incurre en alucinaciones...
        [Contexto Técnico]
        {contexto}

        [Pregunta del Operador]
        {pregunta}

        [Respuesta Propuesta]
        {respuesta}

        Devuelve la evaluación estructurada según el esquema solicitado.
        """
        return self.chain_evaluador.invoke(prompt_eval)

    def _ejecutar_triaje(self, mensaje: str) -> Dict:
        """Realiza el triaje y devuelve el diccionario con decisión, urgencia y motivo."""
        salida: TriajeOperativoOut = self.chain_triaje.invoke([
            SystemMessage(content=PROMPT_TRIAJE_OPERATIVO),
            HumanMessage(content=mensaje)
        ])
        return salida.model_dump()

    def _resolver_con_rag(self, pregunta: str) -> Dict:
        """Ejecuta la recuperación, reranking, generación y validación."""
        if not self.retriever:
            return {
                "respuesta": "No cuento con registros técnicos suficientes...",
                "citaciones": [],
                "rag_exito": False,
                "motivo_fallo": "Búsqueda vectorial no inicializada."
            }

        # 1) Recuperación
        query = f"query: {pregunta}" if REQUIERE_PREFIJO_E5 else pregunta
        candidatos = self.retriever.invoke(query)

        # 2) Reranking
        filtrados = self._ejecutar_reranking(pregunta, candidatos, top_n=3)
        if not filtrados:
            return {
                "respuesta": "No cuento con registros técnicos suficientes...",
                "citaciones": [],
                "rag_exito": False,
                "motivo_fallo": "No se encontraron documentos relevantes."
            }

        # 3) Ensamblar contexto y generar respuesta
        contexto, citas = self._ensamblar_contexto(filtrados)
        respuesta_ia = self.document_chain.invoke({
            "input": pregunta,
            "context": filtrados   # create_stuff_documents_chain espera una lista de documentos
        })

        if "no cuento con registros técnicos" in respuesta_ia.lower():
            return {
                "respuesta": respuesta_ia,
                "citaciones": [],
                "rag_exito": False,
                "motivo_fallo": "La base de conocimientos no cubre el procedimiento."
            }

        # 4) Guardrail de alucinaciones
        evaluacion = self._validar_respuesta(pregunta, contexto, respuesta_ia)
        if not evaluacion.fiel_al_contexto:
            logger.warning(f"Bloqueada respuesta infiel: {evaluacion.justificacion}")
            return {
                "respuesta": "No cuento con registros técnicos suficientes...",
                "citaciones": [],
                "rag_exito": False,
                "motivo_fallo": f"Alucinación detectada: {evaluacion.justificacion}"
            }

        return {
            "respuesta": respuesta_ia,
            "citaciones": citas,
            "rag_exito": True
        }

    # ---------- NODOS DEL GRAFO ----------
    def nodo_condensar_pregunta(self, state: AgentState) -> Dict:
        """Reescribe la pregunta teniendo en cuenta el historial."""
        historial = state.get("messages", [])
        ultima = state["pregunta"]
        if len(historial) <= 1:
            return {"pregunta_condensada": ultima}

        prompt_memoria = ChatPromptTemplate.from_messages([
            ("system", (
                "Dada la siguiente conversación histórica en un entorno industrial (OCI) y una nueva pregunta de seguimiento, "
                "reescríbela para que sea una pregunta independiente y clara. Debe contener de forma explícita nombres de pozos, "
                "equipos, herramientas o incidentes de seguridad mencionados anteriormente. NO respondas la consulta, solo redáctala de nuevo de forma técnica."
            )),
            ("placeholder", "{chat_history}"),
            ("human", "Pregunta de seguimiento: {input}")
        ])
        cadena = prompt_memoria | self.llm
        respuesta = cadena.invoke({
            "chat_history": historial[:-1],
            "input": ultima
        })
        return {"pregunta_condensada": respuesta.content}

    def nodo_triaje(self, state: AgentState) -> Dict:
        logger.info("[NODO: Triaje] Analizando riesgos de la consulta...")
        resultado = self._ejecutar_triaje(state["pregunta_condensada"])
        return {"triaje": resultado}

    def nodo_auto_resolver(self, state: AgentState) -> Dict:
        logger.info("[NODO: Auto-Resolver] Consultando manuales técnicos...")
        resultado = self._resolver_con_rag(state["pregunta_condensada"])
        return {
            "respuesta_RAG": resultado["respuesta"],
            "citaciones": resultado["citaciones"],
            "rag_exito": resultado["rag_exito"],
            "accion_final": "RESOLUCIÓN_AUTOMÁTICA" if resultado["rag_exito"] else "ESCALADO"
        }

    def nodo_pedir_info(self, state: AgentState) -> Dict:
        logger.info("[NODO: Solicitar Detalles] Formulando plantilla...")
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

    def nodo_alertar_supervisor(self, state: AgentState) -> Dict:
        logger.info("[NODO: Alerta Supervisor] Protocolo HITL...")
        resultado = self._resolver_con_rag(state["pregunta_condensada"])
        propuesta = resultado["respuesta"] if resultado["rag_exito"] else "No se encontraron manuales específicos para este escenario crítico."

        alerta = (
            f"**ALERTA DE SEGURIDAD / ESCENARIO CRÍTICO DETECTADO**\n"
            f"**Urgencia**: {state['triaje'].get('urgency', 'ALTA')}\n"
            f"**Motivo**: {state['triaje'].get('motivo', 'Riesgo Técnico')}\n"
            f"----------------------------------------------------------------------\n"
            f"**Recomendación Asistida por IA (Basada en datos)**:\n"
            f"{propuesta}\n"
            f"----------------------------------------------------------------------\n"
            f"**PROTOCOLO HUMAN-IN-THE-LOOP (HITL)**:\n"
            f"Esta consulta involucra operaciones con riesgos potenciales o decisiones de alta prioridad.\n"
            f"**Supervisor asignado**: Por favor evalúe el estado climático actual, el equipo de protección (EPP) y valide físicamente el procedimiento antes de autorizar la maniobra en campo.\n\n"
            f"**[ ] Aprobar Operación   |   [ ] Reconducir Protocolo   |   [ ] Solicitar Verificación de Campo**"
        )
        return {
            "respuesta_RAG": alerta,
            "citaciones": resultado["citaciones"],
            "rag_exito": resultado["rag_exito"],
            "accion_final": "ALERTA_SUPERVISOR_HITL"
        }

    # ---------- ARISTAS CONDICIONALES ----------
    def arista_decision_triaje(self, state: AgentState) -> str:
        decision = state["triaje"]["decision"]
        mapping = {
            "AUTO_RESOLVER": "ir_a_rag",
            "PEDIR_INFO": "ir_a_detalles",
            "ALERTAR_SUPERVISOR": "ir_a_alerta"
        }
        return mapping[decision]

    def arista_evaluacion_rag(self, state: AgentState) -> str:
        if state["rag_exito"]:
            return "finalizar"
        # Si falló el RAG y la consulta es sensible, escalar a supervisor
        palabras_criticas = ["riesgo", "accidente", "emergencia", "pulling", "tensión", "daño", "parada", "tormenta", "viento"]
        if any(p in state["pregunta_condensada"].lower() for p in palabras_criticas):
            logger.info("Fallo de RAG en tema sensible. Escalando a Supervisor.")
            return "ir_a_alerta"
        return "ir_a_detalles"

    # ---------- COMPILACIÓN DEL GRAFO ----------
    def _compilar_grafo(self):
        """Construye y compila el grafo LangGraph."""
        workflow = StateGraph(AgentState)

        # Nodos
        workflow.add_node("condensar_pregunta", self.nodo_condensar_pregunta)
        workflow.add_node("triaje", self.nodo_triaje)
        workflow.add_node("auto_resolver", self.nodo_auto_resolver)
        workflow.add_node("pedir_info", self.nodo_pedir_info)
        workflow.add_node("alertar_supervisor", self.nodo_alertar_supervisor)

        # Flujo principal
        workflow.add_edge(START, "condensar_pregunta")
        workflow.add_edge("condensar_pregunta", "triaje")

        workflow.add_conditional_edges(
            "triaje",
            self.arista_decision_triaje,
            {
                "ir_a_rag": "auto_resolver",
                "ir_a_detalles": "pedir_info",
                "ir_a_alerta": "alertar_supervisor"
            }
        )

        workflow.add_conditional_edges(
            "auto_resolver",
            self.arista_evaluacion_rag,
            {
                "finalizar": END,
                "ir_a_alerta": "alertar_supervisor",
                "ir_a_detalles": "pedir_info"
            }
        )

        workflow.add_edge("pedir_info", END)
        workflow.add_edge("alertar_supervisor", END)

        self.grafo = workflow.compile()
        logger.info("Grafo LangGraph compilado correctamente.")

    # ---------- MÉTODO PÚBLICO ----------
    def procesar_consulta(self, pregunta: str, historial: List[BaseMessage] = None) -> Dict:
        """Punto de entrada para el usuario. Retorna el estado final."""
        if historial is None:
            historial = []
        inputs = {
            "pregunta": pregunta,
            "messages": historial + [HumanMessage(content=pregunta)]
        }
        return self.grafo.invoke(inputs)

