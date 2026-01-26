# Context Engine Inteligente para Marketplace (Mercado Libre)

Prototipo de un **Context Engine** que unifica el contexto de ítems de un marketplace (ítem + vendedor + señales de “salud” del ítem) para que un agente/LLM pueda tomar mejores decisiones.

Este proyecto implementa:
1) **Entidad 360 / Feature Engineering** (perfil denso del ítem)
2) **Retrieval inteligente** (búsqueda semántica + filtro duro de precio)
3) **Insights en tiempo real con GenAI** (LLM genera un JSON estable)

---

## ✅ Qué entrega este repo (Bloques del Challenge)

### Bloque 1 — Modelado Entidad 360 (Feature Engineering)
Se transforma el dataset crudo a un perfil más “denso” y se calculan métricas de salud del ítem, por ejemplo:
- **stock_ratio**
- **sell_through**
- **señales de tags** (normalización/cantidad), y otras features ligeras

📌 Implementado en: `src/features.py` (y usado desde el notebook)

### Bloque 2 — Intelligent Retrieval (Embeddings + filtros)
Dada una búsqueda del usuario (ej: *“Busco una laptop para edición de video que sea económica”*), el sistema recupera items combinando:
- **Filtro duro**: `max_price`
- **Filtro blando**: similitud semántica por embeddings (`score`)

Se guardan artifacts en `.pkl` para reusar sin recalcular.

📌 Implementado en: `notebooks/01_generate_data.ipynb`  
📌 Artifacts: `artifacts/retrieval_artifacts.pkl`, `artifacts/retrieval_artifacts_laptops.pkl` *(generados localmente)*

### Bloque 3 — Real-time Insights & Summarization (GenAI)
Con el contexto recuperado, un LLM (OpenAI) genera una **ficha comparativa** en **JSON estable**:
- `comparative_summary`
- `top_recommendation` (+ reason)
- `risk_alerts`
- `market_insight`

📌 Implementado en: `notebooks/01_generate_data.ipynb`  
📌 Output ejemplo: `notebooks/insights_block3.json`

### Bloque 4 — Arquitectura (MCP & Escalabilidad) *(pendiente / opcional)*
Diagrama y diseño de integración con un servidor MCP y estrategia para mantener reputación del vendedor actualizada sin re-indexar todo.

<img width="1178" height="432" alt="image" src="https://github.com/user-attachments/assets/cbca8233-6eed-44f2-9b4f-54d64c7c03d6" />

Pregunta: ¿Cómo asegurarías que los datos del vendedor estén actualizados sin re-indexar?

Respuesta: Utilizaría una estrategia de consulta híbrida en el servidor MCP. Mantengo el índice de vectores para la búsqueda semántica (que es estática), pero creo una 'Tool' específica que consulta directamente el dataset de vendedores o una API en el momento en que el agente lo solicita. Así, el agente recibe la reputación 'en vivo' recuperada por ID, sin necesidad de generar nuevos embeddings para todo el dataset.

---

## Estructura del repositorio

context-engine-crewai/
├─ artifacts/ # artifacts generados (pkl/parquet)
├─ data/ # datos
├─ notebooks/
│ ├─ 01_generate_data.ipynb # pipeline principal Bloques 1–3
│ ├─ 02_demo_crewai.ipynb 
│ └─ insights_block3.json # salida JSON del Bloque 3
├─ src/
│ └─ features.py # feature engineering + helpers
├─ requirements.txt
├─ .env.example
└─ README.md
