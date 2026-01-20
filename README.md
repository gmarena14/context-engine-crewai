# context-engine-crewai

Prototype of a Context Engine using CrewAI:
- Synthetic dataset generator (100k items + search events)
- 360 item profiles + health metrics
- Semantic retrieval (embeddings + FAISS) with hard budget filter
- GenAI comparative card output as strict JSON

## How to run (local)
```bash
pip install -r requirements.txt
python src/run.py --query "laptop gamer i7 16gb hasta 3000"
