# MIT AI Risk Mapping Tool

Paste a description of an AI project and get back the closest matching risk
domains and subdomains from the MIT AI Risk taxonomy. The tool uses sentence
embeddings to find semantically similar risks, visualizes the match strength,
explains why each risk applies, and can audit many projects at once from a CSV.

## What it does

The app combines a vector matching core with three layers on top of it:

1. **Semantic risk mapping.** Your description is embedded with a
   sentence-transformers model and compared by cosine similarity against every
   risk definition in the MIT taxonomy. The closest matches are returned with
   their similarity scores.
2. **LLM rationale engine.** The top matches plus your description are sent to a
   local Ollama model, which explains in two or three sentences why each risk
   applies, grounded in the official taxonomy definition. The vector layer stays
   the source of truth. If Ollama is not running, the app degrades gracefully and
   still shows the vector matches.
3. **Interactive radar chart.** A Plotly radar maps the similarity scores across
   the top subdomains so you can see at a glance how strongly the project relates
   to each risk.
4. **Bulk CSV audit.** Upload a CSV of project descriptions, pick the text
   column, and download a report of the top matches and scores for every row. One
   shared model is reused across all rows so bulk runs stay fast.

## Project layout

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI that ties everything together |
| `nlp_mapper.py` | Core vector engine: loads the taxonomy, embeds definitions, finds top risks |
| `llm_evaluator.py` | Rationale engine that calls a local Ollama model |
| `visualizations.py` | Builds the Plotly radar chart |
| `bulk_processor.py` | Runs many descriptions through one shared mapper and flattens the report |
| `report_builder.py` | Builds the downloadable CSV and Markdown report for a single analysis |
| `api.py` | FastAPI REST service exposing the mapper over HTTP for programmatic use |
| `evaluate_mapper.py` | Scores the mapper against a gold standard and reports scikit-learn metrics |
| `data/mit_taxonomy.json` | The flattened MIT AI Risk taxonomy used for matching |
| `data/gold_standard.csv` | Curated, class balanced set of real world AI incidents for evaluation |
| `convert_taxonomy.py` | Helper that builds the JSON taxonomy from the source spreadsheet |

## Setup

Requires Python 3.9 or newer.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Optional: enable the rationale engine

The rationale layer uses a local [Ollama](https://ollama.com) model, so no API
key is needed. Install Ollama, then pull and serve a model:

```bash
ollama pull llama3.2
ollama serve
```

You can point the app at a different model or host with environment variables
(see `.env.example`):

```bash
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1
```

If Ollama is not running, everything else still works. The app simply shows the
vector matches without the written rationale.

## How to run

```bash
streamlit run app.py
```

Then open the URL it prints (default http://localhost:8501).

## Usage

1. Paste an AI project description into the text box and click **Analyze Risks**.
2. Review the top matching domains and subdomains with their cosine similarity
   scores and the radar chart.
3. Read the rationale for why each risk applies (requires Ollama).
4. To audit many projects, scroll to **Bulk CSV Audit**, upload a CSV, choose the
   description column, run the audit, and download the report.

## How matching works

`nlp_mapper.py` loads the taxonomy and pre-computes a normalized embedding for
every risk definition once at startup. For a query it embeds the input text and
ranks all definitions by cosine similarity, returning the highest scoring
matches. Because the definition embeddings are cached on the loaded model, only
the query has to be embedded per request.

## REST API

For programmatic and citable use, `api.py` exposes the mapper over HTTP with
FastAPI. The embedding model is loaded once at startup and reused across
requests.

```bash
uvicorn api:app --reload
```

Then open http://localhost:8000/docs for interactive, self documenting
endpoints. The main endpoint accepts a JSON description and returns the top
matches with their cosine scores:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"description": "A health app that predicts patient outcomes.", "top_n": 3}'
```

A `GET /health` endpoint reports whether the model is loaded, which is useful for
uptime checks.

## Evaluating accuracy

`evaluate_mapper.py` measures how well the vector mapper recovers the correct MIT
risk domain for a curated, class balanced gold standard of real world AI
incidents (`data/gold_standard.csv`). It treats the task as single label
classification and reports scikit-learn precision, recall, and F1 (macro and
weighted) plus top 1 and top k accuracy, so the results are citable and
reproducible.

```bash
python evaluate_mapper.py
```
