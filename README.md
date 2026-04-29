# Integrated Hospitality + Retail AI Suite

This repository now hosts one integrated Streamlit application with:
- Agentic AI retail report generation (2-agent workflow)
- Generative AI hospitality concept creator (text + image)
- DevOps-ready deployment assets (Docker + Kubernetes + AWS)

Deployment stack used in this integrated project:
- Docker (containerization)
- Kubernetes (orchestration and service exposure)
- AWS (cloud hosting and deployment target, including EC2/ECR and EKS-based workflow)

## App Modules

- Main page (`app.py`): Agentic retail research workflow
- Streamlit page (`pages/2_Hospitality_GenAI.py`): Hospitality concept generator
- Shared hospitality backend module (`src/genai_module/`)

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
- Docker
- Kubernetes
- AWS (EC2, ECR, EKS workflow)

## Project Structure

```text
integrated-project/
|-- app.py
|-- pages/
|   |-- 2_Hospitality_GenAI.py
|-- src/
|   |-- genai_module/
|   |   |-- config/
|   |   |-- database/
|   |   |-- services/
|   |   |-- utils/
|-- k8s/
|   |-- deployment.yaml
|   |-- service.yaml
|-- legacy/
|   |-- devops_fastapi_demo/
|-- knowledge_base/
|-- vector_store/
|-- internal_repository/
|-- docs/
|-- Dockerfile
|-- requirements.txt
|-- .env.example
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

Hospitality module keys:
- `GROQ_API_KEY`
- `HUGGINGFACE_API_KEY`

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

## AWS Deploy (Docker + Kubernetes)

1. Build and push image:
```powershell
docker build -t <dockerhub-user>/integrated-streamlit-app:latest .
docker push <dockerhub-user>/integrated-streamlit-app:latest
```

2. Update image name in `k8s/deployment.yaml`.

3. Create secret:
```powershell
kubectl create secret generic integrated-app-secrets `
  --from-literal=OPENAI_API_KEY="..." `
  --from-literal=GOOGLE_API_KEY="..." `
  --from-literal=SERPER_API_KEY="..." `
  --from-literal=GEMINI_API_KEY="..." `
  --from-literal=HUGGINGFACE_API_KEY="..."
```

4. Deploy:
```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
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
