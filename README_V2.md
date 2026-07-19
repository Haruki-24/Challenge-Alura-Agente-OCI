
#  Agente OCI: Asistente operativo para Control de Operaciones Industriales
## De Proyecto Piloto a Arquitectura Desacoplada Enterprise (v2.0)
---
Este repositorio contiene la evolución avanzada del **Agente OCI**, diseñado específicamente para la coordinación y mitigación de riesgos en la gestión operativa de la industria petrolera.

Lo que comenzó como una solución para un desafío propuesto por **Alura Latam** — enfocado en centralizar documentación corporativa, y así poder brindar respuestas y recomendaciones de acuerdo a la base de conocimientos — se ha transformado aquí en un entorno experimental de nivel industrial. El propósito de esta versión 2.0 es aplicar y demostrar **competencias profesionales reales de Ingeniería de Software y MLOps**, migrando de un script unificado (monolítico) a una **arquitectura de microservicios desacoplada** diseñada para escalar en la nube (Google Cloud Platform) y operar en canales de mensajería en campo (Telegram)

### 🔍 Accesos Rápidos del Proyecto
*   🚀 **[Demo del MVP Inicial en Streamlit Cloud](https://agente-oil-and-gas-oci.streamlit.app/)**
*   📄 **[Documentación del Piloto Original (Readme v1.0)](https://github.com/Haruki-24/Challenge-Alura-Agente-OCI/blob/main/README.md)**

---

## 🛠️ La Solución de Campo: ¿Qué hace el Agente OCI?

Aprovechando la experiencia real en el sector energético y las complejidades lógicas en operaciones de pozos, pulling y normativas HSE (Salud, Seguridad y Medio Ambiente), el agente opera bajo un flujo de trabajo inteligente (Agentic Workflow) respaldado por 7 manuales técnicos de alta fidelidad redactados para la empresa ficticia OCI:

 - **Clasificación y Triaje Dinámico**: Analiza y prioriza las solicitudes del operador según su urgencia e intención técnica.
 - **Resolución Autónoma (RAG)**: Recupera respuestas precisas desde una base de datos vectorial con manuales de mantenimiento e inventarios de EPP.
 - **Control de Alucinaciones**: Validación semántica en dos capas para garantizar que ninguna respuesta ponga en riesgo la seguridad física en el campo.
- **Human-in-the-loop (HITL)**: El asistente detiene su flujo autónomo y escala el caso requiriendo la validación obligatoria de un supervisor ante maniobras de alto riesgo en campo.

---

## 🚀 Evolución del Agente: Piloto vs. Producción Real

El objetivo de esta nueva versión es romper con el esquema tradicional del piloto —restringido únicamente al entorno web— para poder explotar el potencial del agente de forma práctica durante las operaciones en campo, añadiendo uno de los canales más comunes y de mayor acceso como es Telegram. 

Para lograrlo, el agente ha sido completamente rediseñado bajo una **arquitectura desacoplada de microservicios dedicada a la gestión de riesgos e inteligencia RAG en campo**. A través de dos componentes independientes que interactúan mediante APIs síncronas, se reduce drásticamente la deuda técnica y se prepara el terreno para despliegues Serverless en GCP e integraciones de misión crítica (Web HUD y Telegram mediante n8n). Así, se consolida como un asistente operativo virtual robusto y adaptado a las exigencias reales de la industria de Oil & Gas (bajo el contexto de la empresa ficticia OCI - Operations Control Industrial).


```
Challenge-Alura-Agente-OCI/
├── backend/       # Microservicio API (Cerebro del Agente - FastAPI + LangGraph)
└── frontend/      # Interfaz de Usuario Avanzada (HUD Industrial - Streamlit)
```
---

## 📐 Diseño Arquitectónico del Sistema
```
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
```
---

## 📂 Estructura del Repositorio

El proyecto se divide de manera limpia en:
 - **`backend/`**: Contiene el core del agente (`core/rag_agent_gcp.py`) que maneja los estados del grafo. Utiliza FastAPI (`main.py`) para exponer los endpoints de consulta que alimentarán tanto a la web como a los webhooks de Telegram.

 - **`frontend/`**: Aloja una interfaz web de estilo industrial denominada *AGENTE-OCI*, construida en Streamlit (`app.py`), diseñada específicamente para la auditoría de procesos por parte de los supervisores e ingenieros de oficina.

 - **`data/`** & **`faiss_index_oci/`**: Los documentos técnicos y los índices vectoriales serializados que sirven de conocimiento al agente.

```text
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
```

---

## ⚡ ¿Por qué este diseño?

* **Desacoplamiento Estricto (Frontend/Backend)**: El frontend es extremadamente ligero; no carga modelos de PyTorch, Hugging Face ni bases vectoriales en memoria. Esto permite desplegarlo con recursos mínimos ($256\text{ MB}$ de RAM), mientras que el backend procesa toda la computación pesada ($4\text{ GB}$ de RAM) de forma aislada.

* **Memoria Conversacional con Condensación**: El agente evalúa los turnos previos de conversación para reconstruir la consulta del operador antes de enviarla a la base de datos vectorial, garantizando que el contexto de los pozos e instalaciones nunca se pierda.

* **Flujo de Triaje de Seguridad (HSE Guardrails)**: Gracias a LangGraph, las consultas de alta criticidad (vientos severos, tormentas, maniobras peligrosas) disparan automáticamente alertas de supervisor bloqueando respuestas automatizadas y forzando protocolos Human-in-the-Loop (HITL).

* **RAG en Dos Pasos (Precision Reranking)**: El backend no confía ciegamente en la búsqueda por similitud. Recupera $k=8$ fragmentos y los filtra semánticamente con un modelo Cross-Encoder local, conservando únicamente el Top 3 con mayor peso real.

---

## 🚀 Guía de Ejecución Local (Paso a Paso)

### 1. Configurar las Variables de Entorno

Crea un archivo .env en la raíz del repositorio con tu clave de desarrollo de Gemini:
```text
GEMINI_API_KEY="tu_api_key_de_gemini"
```

### 2. Levantar el Backend (FastAPI)

Abre una terminal, posicionate en la carpeta del backend, crea el entorno virtual e instala sus dependencias pesadas:
```bash
cd backend
python -m venv venv
# Activar entorno (Windows: venv\Scripts\activate | Unix: source venv/bin/activate)
pip install -r requirements.txt
```

Inicia el servidor de API en el puerto 8000:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Puedes validar que el motor cognitivo está arriba ingresando a http://localhost:8000/docs para visualizar la documentación Swagger interactiva.

### 3. Levantar el Frontend (Streamlit HUD)

Abre una segunda terminal, muévete a la carpeta del frontend, inicializa un entorno virtual ligero e instala sus dependencias:
```bash
cd frontend
python -m venv venv
# Activar entorno (Windows: venv\Scripts\activate | Unix: source venv/bin/activate)
pip install -r requirements.txt
```

Ejecuta la interfaz de usuario con temática industrial:
```python
streamlit run app.py
```

La interfaz del operador se abrirá automáticamente en tu navegador en http://localhost:8501.

---

### 📈 Próximos pasos Deploy en GCP

Como proyecto de portafolio, el roadmap técnico contempla el despliegue serverless de esta arquitectura en Google Cloud Platform (GCP) durante un período controlado para demostración técnica:

- **Google Cloud Storage (GCS)**: Traslado de la carpeta /data y el índice /faiss_index_oci a un bucket privado, desacoplando los archivos del código del contenedor.

- **Google Cloud Run (Serverless Backend)**: Despliegue del backend mediante Docker con una asignación de $4\text{ GB}$ de RAM y $2\text{ vCPUs}$ para tolerar la carga de los Cross-Encoders locales de Hugging Face de forma gratuita bajo demanda (escala a 0 instancias si no recibe peticiones).

- **Streamlit Cloud / Cloud Run (Frontend)**: Despliegue ligero de la UI de monitoreo con bajo consumo de infraestructura.

- **n8n Cloud + Telegram API**: Creación de un flujo que consuma el endpoint /chat del Backend en Cloud Run para responder instantáneamente al canal de chat de los operadores de OCI en campo.




