# Autonomous Retail Research Agent

An AI-powered market research application built with Streamlit, CrewAI, and LangChain for generating structured retail intelligence reports from live web data and internal knowledge-base heuristics.

The system is designed for academic demos, high-level design discussions, and collaborative report preparation. It combines a multi-agent workflow with a curated retail knowledge base to produce professional market research reports, downloadable text exports, and PDF-ready assets.

## Overview

The application accepts a retail research query, loads internal retail strategy documents from the `knowledge_base/` directory, and runs a sequential CrewAI workflow:

1. `Retail Researcher`
   - Performs live web research using Serper.
   - Identifies relevant retail trends, demand signals, category opportunities, and margin logic.
   - Cross-references current findings against internal heuristics from the knowledge base.
   - Distinguishes strong evidence from weak-signal sources.

2. `Report Writer`
   - Converts the research findings into a structured market research report.
   - Produces markdown output suitable for presentation, submission, and export.
   - Supports comparison tables and report artifacts used for visualization and PDF export.

## Key Features

- Multi-agent research pipeline using CrewAI
- Shared `Knowledge_Store` built from files in `knowledge_base/`
- Streamlit interface with query input, report display, and sidebar activity logs
- Real-time retail research with Serper-backed web search
- Structured markdown report generation
- Report persistence through `StorageService`
- TXT export for collaboration and PDF export support for formatted submissions
- Optional generated visuals such as summary banners and charts based on report structure

## Tech Stack

- Python
- Streamlit
- CrewAI
- LangChain
- Serper API
- Google Gemini or OpenAI models via environment configuration
- Matplotlib
- ReportLab
- PyPDF / LangChain document loaders

## Project Structure

```text
Agentic AI Project/
├── app.py
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── knowledge_base/
│   ├── retail_margin_framework.txt
│   ├── retail_trends_2026.txt
│   └── retail_tech_strategy.txt
└── internal_repository/
```

## How It Works

### 1. Knowledge Ingestion

The application reads `.txt` and `.pdf` files from `knowledge_base/` using LangChain document loaders. These files are consolidated into a `KnowledgeStore` object containing:

- target profit-margin heuristics
- 2026 trend keywords
- trusted research domains
- source-quality evaluation rules

This allows the agents to use internal retail frameworks as evaluation benchmarks rather than hard-coded constraints.

### 2. Research Workflow

The `Retail Researcher` first investigates live market conditions for the user query and then validates those findings against the knowledge base. The workflow is designed to:

- prioritize real-world market evidence
- preserve local or market-specific findings
- avoid blindly copying the internal knowledge base
- filter generic blogs and weak-signal sources

### 3. Report Generation

The `Report Writer` transforms the research notes into a professional market research report using markdown. Depending on the report structure returned, the application can also derive:

- comparison tables
- trend alignment charts
- margin charts
- PDF report assets

### 4. Persistence

The `StorageService` saves final markdown output to the `internal_repository/` directory using timestamped filenames. Additional generated assets such as charts and PDFs can also be stored there.

## Setup

### 1. Clone or download the project

```powershell
git clone <your-repo-url>
cd "Agentic AI Project"
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and add your API keys:

```env
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
OPENAI_API_KEY=
GOOGLE_API_KEY=
SERPER_API_KEY=
```

Supported model configurations:

- `MODEL_PROVIDER=gemini` with `MODEL_NAME=gemini-2.5-flash`
- `MODEL_PROVIDER=openai` with `MODEL_NAME=gpt-4o`

### 4. Run the app

```powershell
streamlit run app.py
```

The app will be available locally at:

```text
http://localhost:8501
```

## Example Use Case

Sample query:

```text
High-margin wellness trends for D2C brands in India
```

The app will:

- search live market signals and retail sources
- evaluate findings against retail margin and trend heuristics
- generate a structured report for presentation or submission
- save the final output for later retrieval

## Technologies Used

### Streamlit

Used to create the interactive interface, including:

- query input
- report rendering
- download actions
- sidebar activity logs

### CrewAI

Used to orchestrate the multi-agent workflow with sequential task execution.

### LangChain

Used for loading and preparing knowledge-base documents and supporting the knowledge-ingestion layer.

### Serper API

Used for web search and evidence gathering during live research.

### Gemini / OpenAI

Used as the underlying LLM provider for the retail research and report-writing agents.

### Matplotlib and ReportLab

Used for report visuals and PDF generation.

## Security Notes

- Do not commit `.env` to GitHub.
- Keep API keys private.
- Share `.env.example` with collaborators instead of real credentials.
- Generated reports inside `internal_repository/` may contain output from live research; review them before public submission.

## Future Improvements

- stronger deterministic report templates
- richer charts and visual dashboards
- improved PDF styling for submission-ready formatting
- advanced source attribution and confidence scoring
- optional retrieval-augmented search over a larger retail document set

## License

This project currently has no explicit license. Add one before public release if needed.
