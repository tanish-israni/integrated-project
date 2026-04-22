# Retail Research Agent

An agentic AI market research project built with Streamlit, CrewAI, LangChain, and a local vector database. The app accepts a retail research query, runs a two-agent workflow, retrieves supporting memory from a Chroma vector store, and generates a polished market research report with downloads and visuals.

## Live App

- Streamlit deployment link: `https://autonomous-retail-research-agent-kcb9ujagpgvoaik4ngr67u.streamlit.app/`
- Local app URL: `http://127.0.0.1:8501`

## Final Project Highlights

- Two-agent architecture:
  - `Retail Researcher` gathers and compresses useful market evidence
  - `Report Writer` converts that evidence into a stakeholder-ready report
- Vector database integration using Chroma with persisted local storage in `vector_store/`
- Knowledge base ingestion from `knowledge_base/`
- Report persistence in `internal_repository/`
- PDF export, TXT export, margin chart, and trend-alignment chart generation
- Cleaner submission-ready UI with hidden runtime logging
- Separate runtime log file stored locally in `app_runtime.log`

## Architecture

### Agent 1: Retail Researcher

- Uses Serper-backed web research
- Cross-references findings with the knowledge base
- Uses vector retrieval as supporting memory
- Produces a compact fact pack to keep downstream model usage efficient

### Agent 2: Report Writer

- Takes Agent 1's compact fact pack
- Produces a structured market research report in Markdown
- Supports comparison-table extraction for charts and export assets

### Retrieval Layer

- `knowledge_base/` stores curated retail strategy files
- `vector_store/` stores the persisted Chroma vector database
- `internal_repository/` stores generated report artifacts

## Tech Stack

- Python
- Streamlit
- CrewAI
- LangChain
- Chroma
- Serper API
- Groq / Gemini / OpenAI environment-driven model configuration
- Matplotlib
- ReportLab

## Project Structure

```text
Agentic AI Project/
|-- app.py
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- run_app.bat
|-- .streamlit/
|   |-- config.toml
|-- knowledge_base/
|-- docs/
|   |-- AAI Project Report.pdf
|   |-- HLD_AAI.pdf
|   |-- LLD AgenticAI Final.pdf
|-- internal_repository/
|-- vector_store/
|   |-- chroma.sqlite3
|   |-- <chroma index files>
```

## Project Documents

- `docs/AAI Project Report.pdf`
- `docs/HLD_AAI.pdf`
- `docs/LLD AgenticAI Final.pdf`

## Setup

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file from `.env.example` and set your keys.

Example:

```env
MODEL_PROVIDER=groq
MODEL_NAME=llama-3.3-70b-versatile
EMBEDDING_PROVIDER=gemini
OPENAI_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
SERPER_API_KEY=
```

Supported provider patterns in the current app:

- `MODEL_PROVIDER=groq`
- `MODEL_PROVIDER=gemini`
- `MODEL_PROVIDER=openai`

## Run Locally

### Option 1

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

### Option 2

Double-click:

```text
run_app.bat
```

## How the App Works

1. The app loads curated retail knowledge from `knowledge_base/`.
2. It checks or rebuilds the local vector database in `vector_store/` when needed.
3. Agent 1 retrieves live market evidence and vector-memory support.
4. Agent 2 writes the final market research report.
5. The app saves report files and generated assets to `internal_repository/`.

## Outputs

- Market research report rendered in the app
- Downloadable `.txt` report
- Downloadable `.pdf` report
- Opportunity comparison table
- Margin chart
- Trend alignment chart

## Security Notes

- Do not commit `.env`
- Do not commit real API keys
- `app_runtime.log` is local-only and ignored from Git
- `vector_store/` now includes the committed Chroma database artifacts for project review
