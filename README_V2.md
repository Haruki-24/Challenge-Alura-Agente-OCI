
#  Agente OCI: Asistente operativo para Control de Operaciones Industriales

### De Proyecto Piloto a Arquitectura Desacoplada Enterprise (v2.0)

Este repositorio contiene la evolución avanzada del Agente OCI, un sistema de Inteligencia Artificial corporativo diseñado específicamente para la coordinación y mitigación de riesgos en la gestión operativa de campos petroleros.

A partir de un desafío original propuesto por Alura Latam (consistente en centralizar documentación corporativa en una base de conocimiento conversacional), este proyecto se desprendió de su fase piloto para convertirse en un entorno experimental de nivel industrial. El objetivo actual es aplicar y demostrar habilidades profesionales reales de ingeniería de software y MLOps, migrando de un script unificado a una arquitectura de microservicios desacoplada optimizada para la nube (Google Cloud Platform) y canales de mensajería en campo (Telegram).

utilizando Gemini 3.1 Flash-Lite como motor de razonamiento> Versión Cloud Anterior: Puedes revisar el comportamiento del MVP inicial aquí: Agente OCI Piloto en Streamlit Cloud

### 🛠️ La Solución de Campo: ¿Qué hace el Agente OCI?

Aprovechando la experiencia real en el sector energético y las complejidades lógicas en operaciones de pozos, pulling y normativas HSE (Salud, Seguridad y Medio Ambiente), el agente opera bajo un flujo de trabajo inteligente (Agentic Workflow) respaldado por 7 manuales técnicos de alta fidelidad redactados para la empresa ficticia OCI:

Clasificación y Triaje Dinámico: Analiza y prioriza las solicitudes del operador según su urgencia e intención técnica.

Resolución Autónoma (RAG): Recupera respuestas precisas desde una base de datos vectorial con manuales de mantenimiento e inventarios de EPP.

Control de Alucinaciones: Validación semántica en dos capas para garantizar que ninguna respuesta ponga en riesgo la seguridad física en el campo.

Human-in-the-loop (HITL): El asistente detiene su flujo autónomo y escala el caso requiriendo la validación obligatoria de un supervisor ante maniobras de alto riesgo en campo.

### 🚀 Evolución del Agente: Piloto vs. Producción Real

Este repositorio contiene la versión v2.0 Enterprise de AGENTE OCI, un ecosistema de soporte de Inteligencia Artificial diseñado para la industria de Oil & Gas (bajo el contexto de la empresa ficticia OCI - Operations Control Industrial).

Arquitectura Desacoplada de Microservicios para la Gestión de Riesgos e Inteligencia RAG en Campo
Para demostrar competencias de arquitectura robusta, el proyecto ha sido completamente rediseñado y desacoplado en dos microservicios independientes que interactúan mediante APIs síncronas. Esto permite un escalamiento eficiente, reduce la deuda técnica y sienta las bases para despliegues Serverless en Google Cloud Platform (GCP) e integraciones con n8n y Telegram.

Challenge-Alura-Agente-OCI/
├── backend/       # Microservicio API (Cerebro del Agente - FastAPI + LangGraph)
└── frontend/      # Interfaz de Usuario Avanzada (HUD Industrial - Streamlit)


### 📐 Diseño Arquitectónico del Sistema

                        [ OPERADOR DE CAMPO ]
                       /                     \
                      v                       v
             [ Bot de Telegram ]      [ Dashboard Web HUD ]
                      |                       |
               (Webhook n8n)            (HTTP Requests)
                      \                       /
                       v                     v
┌────────────────────────────────────────────────────────────────────────┐
│                        BACKEND SERVICE (FastAPI)                       │
│                                                                        │
│  1. MEMORIA:                                                           │
│     Reescribe consultas de seguimiento en base al historial del chat.   │
│                                                                        │
│  2. TRIAJE INDUSTRIAL (LangGraph):                                     │
│     Clasifica intenciones: AUTO_RESOLVER | PEDIR_INFO | ALERTAR        │
│                                                                        │
│  3. RAG AVANZADO:                                                      │
│     - Recuperación primaria con FAISS + Embeddings Multilingual-E5.    │
│     - Reranking semántico local con Cross-Encoder MiniLM-L-6-v2.       │
│                                                                        │
│  4. HSE GUARDRAILS:                                                    │
│     Valida alucinaciones y bloquea respuestas infieles.                 │
┌────────────────────────────────────────────────────────────────────────┘


### 📂 Estructura del Repositorio

El proyecto se divide de manera limpia en
 - backend/: Contiene el core del agente (core/rag_agent_gcp.py) que maneja los estados del grafo. Utiliza FastAPI (main.py) para exponer los endpoints de consulta que alimentarán tanto a la web como a los webhooks de Telegram.

 - frontend/: Aloja una interfaz web de estilo militar/industrial denominada PETRO-ASSIST, construida en Streamlit (app.py), diseñada específicamente para la auditoría de procesos por parte de los supervisores e ingenieros de oficina.

 - data/ & faiss_index_oci/: Los documentos técnicos y los índices vectoriales serializados que sirven de conocimiento al agente.


Challenge-Alura-Agente-OCI/
├── backend/                              # ⚙️ Microservicio de Procesamiento y API
│   ├── core/
│   │   └── rag_agent_gcp.py              # Cerebro LangGraph (Memoria + Triaje + RAG + Reranking)
│   ├── main.py                           # Servidor FastAPI (API Endpoints)
│   └── requirements.txt                  # Dependencias de IA (LangChain, FAISS, PyTorch, etc.)
├── data/                                 # 📂 Almacén de Documentación Técnica de OCI (PDFs)
│   ├── Inventario_herramientas_EPP.pdf
│   ├── Mision_vision_y_valores_OCI.pdf
│   ├── Politica_HSE_OCI.pdf
│   ├── Procedimiento_supervisor_de_campo_OCI.pdf
│   ├── Procedimiento_y_protocolos_HSE_OCI.pdf
│   ├── Programa_intervenciones_de_pulling.pdf
│   └── Programa_mantenimiento.pdf
├── faiss_index_oci/                      # 🧠 Base Vectorial local persistida
│   ├── index.faiss
│   └── index.pkl
├── frontend/                             # 🎨 Microservicio de Visualización (HUD Dashboard)
│   ├── assets/                           # Archivos multimedia y recursos CSS
│   ├── app.py                            # Streamlit Dashboard (Tema Técnico Industrial)
│   └── requirements.txt                  # Dependencias ligeras del Cliente Web
├── .env                                  # Variables de entorno (API Keys locales)
├── .gitignore                            # Archivos excluidos del control de versiones
└── README.md                             # Documentación maestra del portafolio


### ⚡ ¿Por qué este diseño?

Desacoplamiento Estricto (Frontend/Backend): El frontend es extremadamente ligero; no carga modelos de PyTorch, Hugging Face ni bases vectoriales en memoria. Esto permite desplegarlo con recursos mínimos ($256\text{ MB}$ de RAM), mientras que el backend procesa toda la computación pesada ($4\text{ GB}$ de RAM) de forma aislada.

Memoria Conversacional con Condensación: El agente evalúa los turnos previos de conversación para reconstruir la consulta del operador antes de enviarla a la base de datos vectorial, garantizando que el contexto de los pozos e instalaciones nunca se pierda.

Flujo de Triaje de Seguridad (HSE Guardrails): Gracias a LangGraph, las consultas de alta criticidad (vientos severos, tormentas, maniobras peligrosas) disparan automáticamente alertas de supervisor bloqueando respuestas automatizadas y forzando protocolos Human-in-the-Loop (HITL).

RAG en Dos Pasos (Precision Reranking): El backend no confía ciegamente en la búsqueda por similitud. Recupera $k=8$ fragmentos y los filtra semánticamente con un modelo Cross-Encoder local, conservando únicamente el Top 3 con mayor peso real.

### 🚀 Guía de Ejecución Local (Paso a Paso)

1. Configurar las Variables de Entorno

Crea un archivo .env en la raíz del repositorio con tu clave de desarrollo de Gemini:

GEMINI_API_KEY="tu_api_key_de_gemini"


2. Levantar el Backend (FastAPI)

Abre una terminal, posicionate en la carpeta del backend, crea el entorno virtual e instala sus dependencias pesadas:

cd backend
python -m venv venv
# Activar entorno (Windows: venv\Scripts\activate | Unix: source venv/bin/activate)
pip install -r requirements.txt


Inicia el servidor de API en el puerto 8000:

uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Puedes validar que el motor cognitivo está arriba ingresando a http://localhost:8000/docs para visualizar la documentación Swagger interactiva.

3. Levantar el Frontend (Streamlit HUD)

Abre una segunda terminal, muévete a la carpeta del frontend, inicializa un entorno virtual ligero e instala sus dependencias:

cd frontend
python -m venv venv
# Activar entorno (Windows: venv\Scripts\activate | Unix: source venv/bin/activate)
pip install -r requirements.txt


Ejecuta la interfaz de usuario con temática industrial:

streamlit run app.py


La interfaz del operador se abrirá automáticamente en tu navegador en http://localhost:8501.

### 📈 Próximos pasos Deploy en GCP

Como proyecto de portafolio, el roadmap técnico contempla el despliegue serverless de esta arquitectura en Google Cloud Platform (GCP) durante un período controlado para demostración técnica:

- Google Cloud Storage (GCS): Traslado de la carpeta /data y el índice /faiss_index_oci a un bucket privado, desacoplando los archivos del código del contenedor.

- Google Cloud Run (Serverless Backend): Despliegue del backend mediante Docker con una asignación de $4\text{ GB}$ de RAM y $2\text{ vCPUs}$ para tolerar la carga de los Cross-Encoders locales de Hugging Face de forma gratuita bajo demanda (escala a 0 instancias si no recibe peticiones).

- Streamlit Cloud / Cloud Run (Frontend): Despliegue ligero de la UI de monitoreo con bajo consumo de infraestructura.

- n8n Cloud + Telegram API: Creación de un flujo que consuma el endpoint /chat del Backend en Cloud Run para responder instantáneamente al canal de chat de los operadores de OCI en campo.




