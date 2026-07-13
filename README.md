*Challenge-Alura-Agente-OCI*
---
# **Agente OCI: Inteligencia Artificial para Control de Operaciones Industriales**
---
Este repositorio contiene el desarrollo de OCI Agent, un agente de Inteligencia Artificial diseñado bajo una arquitectura de flujo de trabajo inteligente (Agentic Workflow). Este asistente operativo y de HSE está diseñado específicamente para coordinar y mitigar riesgos en la gestión operativa de campos petroleros, utilizando Gemini 3.1 Flash-Lite como motor de razonamiento y LangGraph para la orquestación estructurada de estados.

**Version Cloud**: ¡Prueba el agente interactivo directamente en tu navegador! *[Agente OCI en Streamlit Cloud](https://agente-oil-and-gas-oci.streamlit.app/)*

## 🎯 Consigna del Desafío (Alura Latam)
El desafío consistió en desarrollar un Agente de Inteligencia Artificial Corporativo accesible para todos los colaboradores, capaz de centralizar y procesar múltiples formatos de archivo (PDF, Word, Excel, etc.) de diversos dominios organizacionales (RRHH, finanzas, legal, operaciones). El objetivo principal era transformarlo en una base de conocimiento conversacional, unificada y siempre disponible.

## 🛠️ Solución Adoptada: Agente OCI
Aprovechando mi experiencia real en el sector y conociendo de primera mano las complejidades lógicas y operativas en los campos petroleros, decidí evitar un caso genérico y diseñar una infraestructura sumamente realista.

Para ello, creé la empresa ficticia OCI, redactando y estructurando de manera consistente toda su documentación técnica (7 archivos PDF que incluyen planes de mantenimiento, programas de pulling, informes de stock y políticas de HSE). Sobre esta base de conocimiento de alta fidelidad, implementé un asistente técnico y de HSE de última generación que ejecuta un flujo de trabajo inteligente (Agentic Workflow):

- Clasificación Dinámica: Analiza y prioriza las solicitudes de los usuarios internos según su nivel de urgencia e intención.

- Resolución Autónoma (RAG): Resuelve consultas operativas y técnicas directamente desde la base de conocimientos semántica.

- Escalabilidad Inteligente: Determina autónomamente si necesita solicitar más detalles al usuario o si debe escalar el caso directamente al supervisor de área.

**Impacto**: Esta solución reduce drásticamente la carga administrativa y logística en la gestión operativa, permitiendo a los supervisores enfocarse en la toma de decisiones críticas en el campo.

---

## 🚀 Evolución del Proyecto
Este proyecto ha recorrido un ciclo de desarrollo profesional, buscando siempre la robustez y la escalabilidad de nivel industrial:

* **Fase de Prototipado (Desafío Alura):** Implementación inicial del sistema RAG sobre Google Colab para validar la viabilidad técnica de la recuperación de documentos.
* **Fase de Producción Local:** Refactorización hacia una arquitectura modular, integrando LangGraph para la orquestación de estados y validación de seguridad (Guardrails con Cross-Encoder local).
* **Fase de Implementación en la Nube (En proceso):** Optimización del código para su despliegue en servicios cloud, priorizando la persistencia de datos y la gestión de API Keys en entornos de producción seguros.

---
# 🤖 Agente Inteligente de Operaciones y Control Industrial (OCI)

Este repositorio contiene el desarrollo del agente de Inteligencia Artificial para el **Control de Operaciones Industriales (OCI)**. El sistema combina el poder de **LangGraph** para la orquestación de flujos de trabajo basados en estados, **LangChain** para la estructuración y recuperación de información técnica, y **Gemini 3.1 Flash-Lite** como motor principal de razonamiento con validación semántica de alucinaciones.

El asistente operativo está diseñado bajo el concepto de **Human-in-the-loop (HITL)**, proporcionando recomendaciones estructuradas basadas en documentación técnica y administrativa de la empresa, pero deteniendo el flujo para requerir la validación de un supervisor humano ante maniobras de alto riesgo en campo.

---

## 🛠️ Stack Tecnológico
Para garantizar la precisión semántica y la predictibilidad del flujo conversacional, se utilizaron las siguientes tecnologías:
- **Interfaz de Usuario**: [Streamlit](https://streamlit.io/) (Para la construcción rápida y despliegue web de la aplicación interactiva).
- **Orquestación de Agentes:** [LangGraph](https://github.com/langchain-ai/langgraph) (Flujos de trabajo basados en estados de alta predictibilidad).
- **Modelo de Lenguaje (LLM):** Google Gemini 3.1 Flash-Lite (Motor principal de razonamiento rápido y eficiente con salidas estructuradas).
- **Framework de Integración:** [LangChain](https://github.com/langchain/langchain) (Cadenas de combinación, cargadores de documentos y abstracción de prompts).
- **Base de Datos Vectorial:** FAISS (Indexación y persistencia local eficiente).
- **Modelos de Embeddings:** HuggingFace `intfloat/multilingual-e5-small` (Generación semántica local utilizando prefijos asimétricos).
- **Modelo de Reranking:** Cross-Encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (Filtro de precisión semántica).
- **Validación Estructurada:** Pydantic V2 (Para esquemas de triaje y controles de guardrail de fidelidad).
- **Procesamiento de Documentos**: PyMuPDF y Unstructured (Extracción limpia de texto y tablas complejas desde los PDFs).

---

## 🧠 Características Principales

### 1. Triaje Operativo Inteligente
Clasifica de manera asertiva las consultas del usuario en categorías lógicas clave (`AUTO_RESOLVER`, `PEDIR_INFO`, `ALERTAR_SUPERVISOR`), analizando de forma dinámica la urgencia del escenario y justificando su clasificación de manera técnica.

### 2. RAG Avanzado con Reranking (Dos Pasos)
A diferencia de los sistemas RAG tradicionales, este agente utiliza un flujo híbrido de dos capas:
1. **Recuperación:** Extrae los 8 fragmentos preliminares más cercanos de la base de datos de FAISS.
2. **Reranking:** Aplica el modelo Cross-Encoder local para calcular la relevancia real entre la consulta y cada fragmento, seleccionando únicamente el Top 3 con mayor peso semántico. Esto reduce las alucinaciones a cero y elimina fragmentos ruidosos.

### 3. Guardrails de Seguridad (HSE) e HITL
El agente está alineado estrictamente bajo el protocolo **Human-in-the-Loop**. Si detecta un escenario crítico (condiciones climáticas adversas como tormentas o ráfagas de viento mayores a 60km/h, accidentes o mantenimiento con tensión eléctrica), cambia de inmediato a la ruta de alerta, detiene los flujos automáticos de respuesta y exige la validación física en campo del supervisor para mitigar riesgos.

---

# 📐 Arquitectura y Flujo de Trabajo

El flujo conversacional transiciona dinámicamente según el estado de la memoria del agente (`AgentState`):

```
    START([Inicio]) --> Triaje[Nodo: Triaje e Intención]
    
    Triaje -->|Decisión: AUTO_RESOLVER| RAG[Nodo: Auto Resolver RAG]
    Triaje -->|Decisión: PEDIR_INFO| Info[Nodo: Solicitar Detalles]
    Triaje -->|Decisión: ALERTAR_SUPERVISOR| Alerta[Nodo: Alerta Supervisor HITL]
    
    RAG -->|¿RAG con éxito?| Fin([Fin])
    RAG -->|¿RAG falló/alucinó con temas críticos?| Alerta
    RAG -->|¿RAG falló sin contexto crítico?| Info
    
```

## 🖼️ Diagrama del Flujo del Agente
![Flujo del Agente de IA](flujo_agente_oci.png)


# 📂 Estructura del Proyecto

```text
├── data/                                      # Documentos de Conocimiento de OCI
│   ├── Inventario_herramientas_EPP.pdf
│   ├── Mision_vision_y_valores_OCI.pdf
│   ├── Politica_HSE_OCI.pdf
│   ├── Procedimiento_supervisor_de_campo_OCI.pdf
│   ├── Procedimiento_y_protocolos_HSE_OCI.pdf
│   ├── Programa_intervenciones_de_pulling.pdf
│   └── Programa_mantenimiento.pdf
├── faiss_index_oci/                           # Base de datos vectorial FAISS local persistida
│   ├── index.faiss
│   └── index.pkl
├── notebooks/                                 # Entornos de desarollo local Google Colab
│   └── Challenge_Alura_Agente_RAG_V1.ipynb    # Archivo para desarrollo y prototipado
├── .gitignore                                 
├── app.py                                     # Archivo base para interfaz y ejecución de app en streamlit
├── flujo_agente_oci.png                       
├── rag_agent_str.py                           # Archivo logico de RAG para app en streamlit
├── RAG_OCI.py                                 # Archivo principal de AGENTE con consola interactiva
├── README.md                                  # Documentación de este repositorio
└── requirements.txt                           # Archivo de dependencias del proyecto
```

---
## 🌐 Demo Interactiva en la Nube

## ¿Quieres probar el agente sin configurar nada de forma local?

Accede a la aplicación desde tu navegador web:
👉 [Probar Demo en Vivo en Streamlit Cloud](https://agente-oil-and-gas-oci.streamlit.app/) 🚀
---

# 🚀 Requisitos e Instalación Local

Si vas a ejecutar el proyecto de forma local en tu computadora, sigue estos pasos:

### 1. Clonar el repositorio y configurar entorno
```bash
git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio

# Creación de entorno virtual
python -m venv venv

# Activar entorno:
# - En Windows: venv\Scripts\activate
# - En macOS/Linux: source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar las variables de entorno (.env)
Crea un archivo `.env` en la raíz del proyecto y configura tus claves:
```text
GEMINI_API_KEY="tu-api-key-de-gemini"
```

### 3. Ejecutar la Consola Interactiva

* Opcion A:
```bash
python RAG_OCI.py

```
* Opcion B:
```bash
python streamlit run app.py

```


---

# 📓 Alternativa: Ejecución en Google Colab

Este script está completamente optimizado para funcionar tanto de forma local como en la nube de **Google Colab** utilizando la ruta `notebooks/Challenge_Alura_Agente_RAG_V1.ipynb`.

### 1. Instalar dependencias
```python
!pip install -r requirements.txt
```

### 2. Configurar la API Key de forma segura
Para no exponer tu API Key, utiliza la herramienta nativa de Google Colab:
1. Abre el panel de **Secrets**.
2. Agrega una nueva clave **`Gemini-API-key`** (Reemplazar en el archivo `Gemini-GC`")
3. Pega tu API Key en el valor y activa la casilla **Notebook access**.

---

# 📈 Demostración de Resultados Reales (Logs de Consola)

### Test 1: Consulta Procedimental de Mantenimientos Programados
```text
Ingresa tu pregunta ---> ¿Que pozos tienen tareas de Mantenimiento estan porgramadas?
Procesando consulta...
[NODO: Triaje] Analizando riesgos de la consulta...
[NODO: Auto-Resolver] Consultando manuales técnicos...

--- RESULTADO DEL TRIAJE ---
Consulta: "¿Que pozos tienen tareas de Mantenimiento estan porgramadas?"
Categoría asignada: AUTO_RESOLVER
Urgencia priorizada: MEDIANA
Motivo técnico: Consulta informativa sobre la planificación de mantenimiento de los pozos.
Acción adoptada por el Grafo: RESOLUCIÓN_AUTOMÁTICA

--- RESPUESTA Y DIRECTRICES TÉCNICAS ---
Las tareas de mantenimiento programadas de acuerdo con el Programa de Mantenimiento de OCI son:
- Pozo E-501 (Mantenimiento Eléctrico): "Chequeo tablero" programado para el 26/06/2026.
- Pozo M-909 (Mantenimiento Eléctrico): "Medición aislamiento" programado para el 26/06/2026.
- Pozo M-707 (Mantenimiento Mecánico PCP): "Retiro cabezal (Pre-Pulling)" programado para el 27/06/2026.
- Pozo E-808 (Mantenimiento Mecánico AIB): "Reemplazo controlador" programado para el 28/06/2026.

⚠️ ADVERTENCIA DE SEGURIDAD (HSE)
Toda intervención requiere la validación en campo de los elementos de izaje y el uso estricto del EPP reglamentario.
El Supervisor de Campo tiene la decisión final frente a cualquier imprevisto técnico o climático.

--- CITACIONES DE RESPALDO (RAG) ---
  [1] Fuente: 'Programa_mantenimiento.pdf (Pág. 1)'
```

### Test 2: Consulta Específica de Fecha de Intervención (Pulling)
```text
Ingresa tu pregunta ---> ¿Cuando es la intervencion de pulling del pozo P-404?
Procesando consulta...
[NODO: Triaje] Analizando riesgos de la consulta...
[NODO: Auto-Resolver] Consultando manuales técnicos...

--- RESULTADO DEL TRIAJE ---
Consulta: "¿Cuando es la intervencion de pulling del pozo P-404?"
Categoría asignada: AUTO_RESOLVER
Urgencia priorizada: MEDIANA
Motivo técnico: Consulta de fecha del cronograma de intervenciones de pulling del pozo P-404.
Acción adoptada por el Grafo: RESOLUCIÓN_AUTOMÁTICA

--- RESPUESTA Y DIRECTRICES TÉCNICAS ---
La intervención de pulling del pozo P-404 está programada para el 26/06/2026. La tarea involucra falla de varilla y requiere el uso de Grúa Pesada, con una duración estimada de 2 a 5 días a cargo de la Cuadrilla Beta.

⚠️ ADVERTENCIA DE SEGURIDAD (HSE)
El izaje de cargas pesadas en maniobras de pulling de varillas se clasifica como operación de alto riesgo. Se debe corroborar la vigencia de la certificación del operador de grúa antes de iniciar el procedimiento.
El Supervisor de Campo tiene la decisión final frente a cualquier imprevisto técnico o climático.

--- CITACIONES DE RESPALDO (RAG) ---
  [1] Fuente: 'Programa_intervenciones_de_pulling.pdf (Pág. 1)'
```
