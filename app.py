from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


load_dotenv()

APP_TITLE = "Autonomous Retail Research Agent (v2)"
ROOT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge_base"
INTERNAL_REPOSITORY_DIR = ROOT_DIR / "internal_repository"
REPORT_CSS = """
<style>
div[data-testid="stMarkdownContainer"] .report-shell {
    background: linear-gradient(180deg, #fffdf7 0%, #fff8ee 100%);
    border: 1px solid #eadbc8;
    border-radius: 18px;
    padding: 1.6rem 1.4rem;
    box-shadow: 0 10px 30px rgba(101, 67, 33, 0.08);
}
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: #7c2d12;
    letter-spacing: normal;
    word-spacing: normal;
    line-height: 1.25;
}
div[data-testid="stMarkdownContainer"] h1 {
    margin-top: 0.2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #f6b26b;
}
div[data-testid="stMarkdownContainer"] h2 {
    margin-top: 1.5rem;
    margin-bottom: 0.7rem;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: #2f2a24;
    line-height: 1.7;
    letter-spacing: normal;
    word-spacing: normal;
    text-align: left;
}
div[data-testid="stMarkdownContainer"] strong {
    color: #17313e;
    letter-spacing: normal;
    word-spacing: normal;
}
div[data-testid="stMarkdownContainer"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0 1.4rem 0;
}
div[data-testid="stMarkdownContainer"] th {
    background: #17313e;
    color: #fff;
    padding: 0.7rem;
    text-align: left;
}
div[data-testid="stMarkdownContainer"] td {
    border: 1px solid #e7d8c9;
    padding: 0.7rem;
}
</style>
"""


@dataclass
class ComparisonRow:
    product_name: str
    est_margin: str
    trend_alignment: str


@dataclass
class VisualReportAssets:
    comparison_rows: List[ComparisonRow]
    margin_chart_path: Optional[Path]
    alignment_chart_path: Optional[Path]
    summary_image_path: Optional[Path]
    pdf_bytes: bytes
    pdf_path: Path


class StorageService:
    def __init__(self, repository_dir: Path) -> None:
        self.repository_dir = repository_dir
        self.repository_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, report_markdown: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.repository_dir / f"retail_report_{timestamp}.txt"
        file_path.write_text(report_markdown, encoding="utf-8")
        return file_path

    def retrieve_report(self, filename: Optional[str] = None) -> Optional[str]:
        if filename:
            target = self.repository_dir / filename
            if target.exists():
                return target.read_text(encoding="utf-8")
            return None

        reports = sorted(self.repository_dir.glob("retail_report_*.txt"), reverse=True)
        if not reports:
            return None
        return reports[0].read_text(encoding="utf-8")

    def build_report_paths(self, stem: str) -> dict[str, Path]:
        return {
            "txt": self.repository_dir / f"{stem}.txt",
            "pdf": self.repository_dir / f"{stem}.pdf",
            "margin_chart": self.repository_dir / f"{stem}_margin_chart.png",
            "alignment_chart": self.repository_dir / f"{stem}_alignment_chart.png",
            "summary_image": self.repository_dir / f"{stem}_summary.png",
        }

    def save_report_bundle(
        self,
        report_markdown: str,
        pdf_bytes: bytes,
        margin_chart_bytes: Optional[bytes],
        alignment_chart_bytes: Optional[bytes],
        summary_image_bytes: Optional[bytes],
    ) -> dict[str, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"retail_report_{timestamp}"
        paths = self.build_report_paths(stem)

        paths["txt"].write_text(report_markdown, encoding="utf-8")
        paths["pdf"].write_bytes(pdf_bytes)
        if margin_chart_bytes:
            paths["margin_chart"].write_bytes(margin_chart_bytes)
        if alignment_chart_bytes:
            paths["alignment_chart"].write_bytes(alignment_chart_bytes)
        if summary_image_bytes:
            paths["summary_image"].write_bytes(summary_image_bytes)
        return paths


class StreamlitLogger:
    def __init__(self, placeholder) -> None:
        self.placeholder = placeholder

    def reset(self) -> None:
        st.session_state.agent_logs = []
        self.render()

    def write(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.agent_logs.append(f"[{timestamp}] {message}")
        self.render()

    def render(self) -> None:
        logs = st.session_state.get("agent_logs", [])
        self.placeholder.code("\n".join(logs) if logs else "Waiting for a run...", language="text")

    def step_callback(self, step_output) -> None:
        message = self._extract_step_message(step_output)
        if message:
            self.write(message)

    @staticmethod
    def _extract_step_message(step_output) -> Optional[str]:
        if step_output is None:
            return None
        if isinstance(step_output, str):
            return step_output[:400]
        if isinstance(step_output, dict):
            for key in ("message", "thought", "text", "output", "result"):
                value = step_output.get(key)
                if value:
                    return str(value)[:400]
            return str(step_output)[:400]

        for attr in ("message", "thought", "text", "output", "result"):
            value = getattr(step_output, attr, None)
            if value:
                return str(value)[:400]
        return str(step_output)[:400]


@dataclass
class KnowledgeStore:
    source_names: List[str]
    raw_context: str
    target_profit_margins: str
    trend_keywords: List[str]
    trusted_domains: List[str]
    source_quality_rules: str

    def research_brief(self) -> str:
        return (
            "Knowledge Store Brief\n"
            f"- Sources: {', '.join(self.source_names) if self.source_names else 'None'}\n"
            f"- Target Profit Margins: {self.target_profit_margins or 'Not found'}\n"
            f"- 2026 Trend Keywords: {', '.join(self.trend_keywords) if self.trend_keywords else 'Not found'}\n"
            f"- Trusted Domains: {', '.join(self.trusted_domains) if self.trusted_domains else 'Not found'}\n"
            f"- Source Quality Rules: {self.source_quality_rules or 'Not found'}\n"
        )


class KnowledgeIngestion:
    def __init__(self, knowledge_dir: Path) -> None:
        self.knowledge_dir = knowledge_dir
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self) -> KnowledgeStore:
        documents = []
        for pattern, loader_cls in (("*.txt", TextLoader), ("*.pdf", PyPDFLoader)):
            loader = DirectoryLoader(
                str(self.knowledge_dir),
                glob=pattern,
                loader_cls=loader_cls,
                show_progress=False,
                loader_kwargs={"encoding": "utf-8"} if loader_cls is TextLoader else None,
            )
            documents.extend(loader.load())

        source_names = sorted(
            {Path(doc.metadata.get("source", "unknown")).name for doc in documents}
        )
        content_map = {
            Path(doc.metadata.get("source", "unknown")).name: doc.page_content
            for doc in documents
        }
        raw_context = "\n\n".join(
            f"[{name}]\n{content_map[name].strip()}" for name in sorted(content_map)
        )

        margin_text = content_map.get("retail_margin_framework.txt", "")
        trend_text = content_map.get("retail_trends_2026.txt", "")
        tech_text = content_map.get("retail_tech_strategy.txt", "")

        return KnowledgeStore(
            source_names=source_names,
            raw_context=raw_context or "No knowledge files were loaded.",
            target_profit_margins=self._extract_target_profit_margins(margin_text),
            trend_keywords=self._extract_trend_keywords(trend_text),
            trusted_domains=self._extract_trusted_domains(tech_text),
            source_quality_rules=self._extract_source_quality_rules(tech_text),
        )

    @staticmethod
    def _extract_target_profit_margins(text: str) -> str:
        rules = []
        for marker in (
            "Gross Margin Threshold:",
            "High-Margin Sectors:",
            "Low-Margin/High-Volume Sectors:",
            "Key Financial Metric:",
            "Research Directive:",
        ):
            match = re.search(rf"{re.escape(marker)}\s*(.+)", text)
            if match:
                rules.append(f"{marker} {match.group(1).strip()}")
        return "\n".join(rules)

    @staticmethod
    def _extract_trend_keywords(text: str) -> List[str]:
        keywords = []
        marker_map = {
            "Digital-First Personalization:": "digital-first personalization",
            "Subscription Economy:": "subscription economy",
            "Sustainable Premiumization:": "sustainable premiumization",
            "Social Commerce": "social commerce",
            "TikTok/Reels": "short-form video commerce",
            "AI-driven recommendations": "AI-driven recommendations",
            "replenishment subscriptions": "replenishment subscriptions",
            "supply chain transparency": "supply chain transparency",
        }
        lower_text = text.lower()
        for marker, keyword in marker_map.items():
            if marker.lower() in lower_text:
                keywords.append(keyword)
        return keywords

    @staticmethod
    def _extract_trusted_domains(text: str) -> List[str]:
        domain_map = {
            "Gartner Research": "gartner.com",
            "McKinsey Retail & Consumer Goods": "mckinsey.com",
            "Statista": "statista.com",
            "SEC Filings": "sec.gov",
        }
        domains = []
        for marker, domain in domain_map.items():
            if marker in text:
                domains.append(domain)
        return domains

    @staticmethod
    def _extract_source_quality_rules(text: str) -> str:
        match = re.search(r"Research Directive:\s*(.+)", text)
        return match.group(1).strip() if match else ""


def ensure_session_defaults() -> None:
    st.session_state.setdefault("agent_logs", [])
    st.session_state.setdefault("latest_report", "")
    st.session_state.setdefault("latest_report_path", "")
    st.session_state.setdefault("latest_pdf_path", "")
    st.session_state.setdefault("latest_query", "")
    st.session_state.setdefault("knowledge_store", None)
    st.session_state.setdefault("latest_comparison_rows", [])
    st.session_state.setdefault("latest_margin_chart_path", "")
    st.session_state.setdefault("latest_alignment_chart_path", "")
    st.session_state.setdefault("latest_summary_image_path", "")
    st.session_state.setdefault("latest_pdf_bytes", b"")


def validate_environment(provider: str) -> List[str]:
    missing = []
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            missing.append("GOOGLE_API_KEY")
    else:
        if not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")

    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")
    return missing


def build_llm() -> LLM:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        model_name = os.getenv("MODEL_NAME", "gemini-1.5-pro")
        return LLM(
            model=f"gemini/{model_name}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )

    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    return LLM(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
    )


def build_retail_crew(query: str, knowledge_store: KnowledgeStore, logger: StreamlitLogger) -> Crew:
    llm = build_llm()
    search_tool = SerperDevTool()

    knowledge_brief = knowledge_store.research_brief()
    trusted_domains = ", ".join(knowledge_store.trusted_domains) if knowledge_store.trusted_domains else "No preferred domains available"
    trend_keywords = ", ".join(knowledge_store.trend_keywords) if knowledge_store.trend_keywords else "No trend keywords available"

    researcher = Agent(
        role="Retail Researcher",
        goal=(
            "Act like a senior market analyst who starts from the knowledge base, then performs targeted web validation "
            "to find high-margin, high-confidence retail opportunities."
        ),
        backstory=(
            "You are a seasoned retail strategy analyst. Before any web search, you must query the Knowledge_Store and extract "
            "target profit margins, 2026 trend keywords, and trusted domains. You then search specifically with those signals, "
            "cross-reference results against the knowledge base, and filter out weak signals from generic blogs.\n\n"
            f"{knowledge_brief}\n"
            f"Full Knowledge Store Context:\n{knowledge_store.raw_context}"
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        step_callback=logger.step_callback,
    )

    writer = Agent(
        role="Report Writer",
        goal="Create a clean, professional Market Research Report in Markdown for stakeholders.",
        backstory=(
            "You are a technical business writer. You convert the researcher's source-aware findings into a polished report "
            "with strong visual structure. You always include an Executive Summary, H1/H2 headers, bullet lists, and a comparison table."
        ),
        llm=llm,
        verbose=True,
        step_callback=logger.step_callback,
    )

    research_task = Task(
        description=(
            f"User research query: {query}\n\n"
            "Mandatory workflow:\n"
            "1. FIRST, perform a comprehensive web search for the user's query and gather real-time market data, notable brands, demand signals, and growth drivers.\n"
            "2. Identify real-world trends, product categories, and commercial opportunities from current web evidence before applying any internal framework.\n"
            "3. SECOND, cross-reference your findings against the Knowledge_Store and explicitly extract these benchmarks:\n"
            "   - Target Profit Margins\n"
            "   - 2026 Trend Keywords\n"
            "   - Trusted Domain Lists\n"
            f"4. Prioritize findings from these trusted domains when possible: {trusted_domains}\n"
            f"5. Use these trend keywords as evaluation signals where relevant, not as hard filters: {trend_keywords}\n"
            "6. If a trend matches the High-Margin or Growth criteria in the Knowledge_Store, highlight that alignment clearly.\n"
            "7. If a trend is uniquely local or market-specific and supported by live web evidence, report it even if it is not explicitly mentioned in the Knowledge_Store.\n"
            "8. Do not copy-paste from the Knowledge_Store. Use it only as a benchmark to validate, challenge, or contextualize your live search results.\n"
            "9. If a source is a generic blog, mark it as Weak Signal. If a source is from a trusted industry research firm or sec.gov, treat it as Ground Truth.\n"
            "10. Include estimated margin logic, trend-fit notes, validation notes, source quality, and risks.\n\n"
            "Return a structured research brief with these sections:\n"
            "- Knowledge Store Extraction\n"
            "- Verified Opportunities\n"
            "- India or Market-Specific Signals\n"
            "- Weak Signals Rejected\n"
            "- Margin and Trend Logic\n"
            "- Validation Notes\n"
            "- Trusted Sources Used"
        ),
        expected_output=(
            "A detailed list of real-world trends and opportunities with a validation note for each one showing "
            "how it aligns with, extends, or challenges the internal retail frameworks."
        ),
        agent=researcher,
    )

    report_task = Task(
        description=(
            f"Write the final response for the query: {query}\n\n"
            "Format the output exactly as a professional Market Research Report using Markdown.\n"
            "Formatting requirements:\n"
            "- Start with exactly one H1 title\n"
            "- Include an 'Executive Summary' section at the top\n"
            "- Use H2 headers for all major sections with a blank line between sections\n"
            "- Use bulleted lists for insights and recommendations\n"
            "- Include a comparison table with columns: Product Name | Est. Margin | Trend Alignment\n"
            "- The comparison table is mandatory and must contain at least 4 rows\n"
            "- Est. Margin values must be numeric percentages like 42% or 38-45%\n"
            "- Trend Alignment values must be concise labels like Very High, High, Medium, or Low\n"
            "- Preserve source-aware nuance and distinguish Ground Truth from Weak Signal\n"
            "- Avoid excessive bold formatting and do not stretch words with decorative spacing\n"
            "- End with a Sources section\n"
            "- Ensure the final output renders cleanly in Streamlit markdown\n\n"
            "Use this exact section order:\n"
            "# Market Research Report: <Title>\n"
            "## Executive Summary\n"
            "## Market Landscape\n"
            "## Key Opportunities\n"
            "## Opportunity Comparison\n"
            "## Risks and Validation Notes\n"
            "## Recommendations\n"
            "## Sources"
        ),
        expected_output="A final Markdown Market Research Report ready for rendering and download.",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, report_task],
        process=Process.sequential,
        verbose=True,
    )


def ingest_knowledge_store(logger: Optional[StreamlitLogger] = None) -> KnowledgeStore:
    if logger:
        logger.write("Ingesting all files in knowledge_base into the Knowledge_Store.")
    store = KnowledgeIngestion(KNOWLEDGE_BASE_DIR).ingest()
    if logger:
        logger.write(
            f"Knowledge_Store ready with {len(store.source_names)} source file(s): "
            f"{', '.join(store.source_names) if store.source_names else 'none'}."
        )
    return store


def parse_comparison_table(report_markdown: str) -> List[ComparisonRow]:
    table_lines = [line.strip() for line in report_markdown.splitlines() if line.strip().startswith("|")]
    rows: List[ComparisonRow] = []
    if len(table_lines) < 3:
        return rows

    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        rows.append(
            ComparisonRow(
                product_name=cells[0],
                est_margin=cells[1],
                trend_alignment=cells[2],
            )
        )
    return rows


def parse_markdown_sections(report_markdown: str) -> List[tuple[str, List[str]]]:
    sections: List[tuple[str, List[str]]] = []
    current_title = "Report"
    current_lines: List[str] = []
    for raw_line in report_markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return sections


def extract_margin_value(margin_text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", margin_text)
    return float(match.group(1)) if match else 0.0


def alignment_to_score(alignment_text: str) -> int:
    lowered = alignment_text.lower()
    if "very high" in lowered or "strong" in lowered:
        return 5
    if "high" in lowered:
        return 4
    if "medium" in lowered or "moderate" in lowered:
        return 3
    if "low" in lowered:
        return 2
    return 1


def render_chart_to_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buffer.getvalue()


def create_margin_chart(rows: List[ComparisonRow]) -> Optional[bytes]:
    if not rows:
        return None

    labels = [row.product_name[:18] for row in rows[:6]]
    values = [extract_margin_value(row.est_margin) for row in rows[:6]]
    if not any(values):
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#f8f5ef")
    ax.bar(labels, values, color=["#1f6f78", "#3baea0", "#f6b042", "#f08a5d", "#b83b5e", "#6a2c70"][: len(labels)])
    ax.set_title("Estimated Margin by Opportunity", fontsize=14, fontweight="bold", color="#17313e")
    ax.set_ylabel("Estimated Margin (%)")
    ax.set_ylim(0, max(values) + 10)
    ax.set_facecolor("#f8f5ef")
    ax.tick_params(axis="x", rotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return render_chart_to_bytes(fig)


def create_alignment_chart(rows: List[ComparisonRow]) -> Optional[bytes]:
    if not rows:
        return None

    labels = [row.product_name[:16] for row in rows[:6]]
    values = [alignment_to_score(row.trend_alignment) for row in rows[:6]]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#f6f4ff")
    ax.plot(labels, values, marker="o", linewidth=3, color="#4259c1")
    ax.fill_between(labels, values, [0] * len(values), color="#9cb3ff", alpha=0.35)
    ax.set_title("Trend Alignment Score", fontsize=14, fontweight="bold", color="#28315c")
    ax.set_ylabel("Score (1-5)")
    ax.set_ylim(0, 5.5)
    ax.set_facecolor("#f6f4ff")
    ax.tick_params(axis="x", rotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return render_chart_to_bytes(fig)


def extract_executive_summary(report_markdown: str) -> List[str]:
    summary_lines: List[str] = []
    in_section = False
    for line in report_markdown.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("## executive summary"):
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and stripped:
            cleaned = stripped.lstrip("- ").strip()
            if cleaned:
                summary_lines.append(cleaned)
    return summary_lines[:3]


def create_summary_image(query: str, summary_points: List[str]) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="#fff9f0")
    ax.set_facecolor("#fff9f0")
    ax.axis("off")
    ax.text(0.03, 0.9, "Retail Research Snapshot", fontsize=22, fontweight="bold", color="#7c2d12")
    ax.text(0.03, 0.78, query[:90], fontsize=12, color="#444444")
    colors_list = ["#f97316", "#ea580c", "#c2410c"]
    for index, point in enumerate(summary_points or ["Report generated successfully."]):
        y = 0.58 - (index * 0.18)
        ax.add_patch(plt.Rectangle((0.03, y - 0.02), 0.025, 0.05, color=colors_list[index % len(colors_list)], transform=ax.transAxes))
        ax.text(0.08, y, point[:120], fontsize=12, color="#3f3f46", va="center", transform=ax.transAxes)
    ax.text(0.03, 0.08, "Autonomous Retail Research Agent", fontsize=10, color="#9a3412")
    return render_chart_to_bytes(fig)


def markdown_to_paragraphs(report_markdown: str, body_style: ParagraphStyle, heading_style: ParagraphStyle) -> List:
    flowables: List = []
    for raw_line in report_markdown.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):
            continue
        if set(line.replace("|", "").replace("-", "").strip()) == set():
            continue
        if line.startswith("# "):
            flowables.append(Paragraph(line[2:].strip(), heading_style))
        elif line.startswith("## "):
            flowables.append(Paragraph(line[3:].strip(), heading_style))
        elif line.startswith("- "):
            flowables.append(Paragraph(f"&bull; {line[2:].strip()}", body_style))
        else:
            flowables.append(Paragraph(line, body_style))
        flowables.append(Spacer(1, 0.12 * inch))
    return flowables


def build_pdf_report(
    query: str,
    report_markdown: str,
    comparison_rows: List[ComparisonRow],
    summary_image_bytes: Optional[bytes],
    margin_chart_bytes: Optional[bytes],
    alignment_chart_bytes: Optional[bytes],
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#17313e"),
        spaceAfter=12,
        alignment=TA_LEFT,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#7c2d12"),
        spaceAfter=8,
        spaceBefore=10,
        alignment=TA_LEFT,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["BodyText"],
        fontSize=9.5,
        textColor=colors.HexColor("#525252"),
        spaceAfter=10,
    )

    story: List = [
        Paragraph("Autonomous Retail Research Agent", title_style),
        Paragraph(query, meta_style),
        Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y %H:%M')}", meta_style),
        Spacer(1, 0.1 * inch),
    ]

    if summary_image_bytes:
        story.append(Image(io.BytesIO(summary_image_bytes), width=6.7 * inch, height=3.4 * inch))
        story.append(Spacer(1, 0.2 * inch))

    sections = parse_markdown_sections(report_markdown)
    if sections:
        first_section = True
        for section_title, section_lines in sections:
            if not first_section and section_title in {
                "Market Landscape",
                "Opportunity Comparison",
                "Risks and Validation Notes",
                "Sources",
            }:
                story.append(PageBreak())
            first_section = False
            if section_title != "Report":
                story.append(Paragraph(section_title, heading_style))
                story.append(Spacer(1, 0.08 * inch))
            story.extend(markdown_to_paragraphs("\n".join(section_lines), body_style, heading_style))
    else:
        story.extend(markdown_to_paragraphs(report_markdown, body_style, heading_style))

    if comparison_rows:
        story.append(PageBreak())
        story.append(Paragraph("Opportunity Comparison Table", heading_style))
        table_data = [["Product Name", "Est. Margin", "Trend Alignment"]]
        table_data.extend([[row.product_name, row.est_margin, row.trend_alignment] for row in comparison_rows[:8]])
        table = Table(table_data, colWidths=[2.8 * inch, 1.4 * inch, 2.0 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17313e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#d6d3d1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#fff7ed")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.22 * inch))

    if margin_chart_bytes:
        story.append(Paragraph("Estimated Margin Chart", heading_style))
        story.append(Image(io.BytesIO(margin_chart_bytes), width=6.5 * inch, height=3.6 * inch))
        story.append(Spacer(1, 0.18 * inch))

    if alignment_chart_bytes:
        story.append(Paragraph("Trend Alignment Chart", heading_style))
        story.append(Image(io.BytesIO(alignment_chart_bytes), width=6.5 * inch, height=3.6 * inch))

    doc.build(story)
    return buffer.getvalue()


def build_visual_assets(query: str, report_markdown: str, storage_service: StorageService) -> VisualReportAssets:
    comparison_rows = parse_comparison_table(report_markdown)
    margin_chart_bytes = create_margin_chart(comparison_rows)
    alignment_chart_bytes = create_alignment_chart(comparison_rows)
    summary_image_bytes = create_summary_image(query, extract_executive_summary(report_markdown))
    pdf_bytes = build_pdf_report(
        query=query,
        report_markdown=report_markdown,
        comparison_rows=comparison_rows,
        summary_image_bytes=summary_image_bytes,
        margin_chart_bytes=margin_chart_bytes,
        alignment_chart_bytes=alignment_chart_bytes,
    )
    saved_paths = storage_service.save_report_bundle(
        report_markdown=report_markdown,
        pdf_bytes=pdf_bytes,
        margin_chart_bytes=margin_chart_bytes,
        alignment_chart_bytes=alignment_chart_bytes,
        summary_image_bytes=summary_image_bytes,
    )
    return VisualReportAssets(
        comparison_rows=comparison_rows,
        margin_chart_path=saved_paths.get("margin_chart"),
        alignment_chart_path=saved_paths.get("alignment_chart"),
        summary_image_path=saved_paths.get("summary_image"),
        pdf_bytes=pdf_bytes,
        pdf_path=saved_paths["pdf"],
    )


def run_research(
    query: str,
    logger: StreamlitLogger,
    storage_service: StorageService,
    knowledge_store: KnowledgeStore,
) -> Path:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    missing_vars = validate_environment(provider)
    if missing_vars:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing_vars)}")

    logger.write("Building CrewAI agents with the preloaded Knowledge_Store.")
    logger.write(f"Trusted domains loaded: {', '.join(knowledge_store.trusted_domains) or 'none'}")
    logger.write("Starting sequential execution: Retail Researcher -> Report Writer.")
    crew = build_retail_crew(query, knowledge_store, logger)
    result = crew.kickoff()

    final_report = result.raw if hasattr(result, "raw") else str(result)
    visual_assets = build_visual_assets(query, final_report, storage_service)
    report_path = visual_assets.pdf_path.with_suffix(".txt")
    logger.write(f"Saved final Markdown report exactly as rendered to {report_path.name}.")
    logger.write(f"Built PDF export and visual assets for {visual_assets.pdf_path.name}.")

    st.session_state.latest_report = final_report
    st.session_state.latest_report_path = str(report_path)
    st.session_state.latest_pdf_path = str(visual_assets.pdf_path)
    st.session_state.latest_query = query
    st.session_state.latest_comparison_rows = visual_assets.comparison_rows
    st.session_state.latest_margin_chart_path = str(visual_assets.margin_chart_path) if visual_assets.margin_chart_path else ""
    st.session_state.latest_alignment_chart_path = str(visual_assets.alignment_chart_path) if visual_assets.alignment_chart_path else ""
    st.session_state.latest_summary_image_path = str(visual_assets.summary_image_path) if visual_assets.summary_image_path else ""
    st.session_state.latest_pdf_bytes = visual_assets.pdf_bytes
    return report_path


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=":bar_chart:", layout="wide")
    ensure_session_defaults()

    storage_service = StorageService(INTERNAL_REPOSITORY_DIR)
    st.markdown(REPORT_CSS, unsafe_allow_html=True)

    st.title(APP_TITLE)
    st.caption("CrewAI + Streamlit retail research with a shared Knowledge_Store and professional Markdown reporting.")

    with st.sidebar:
        st.subheader("Agent Activity")
        log_placeholder = st.empty()
        logger = StreamlitLogger(log_placeholder)
        logger.render()

        if st.session_state.knowledge_store is None:
            st.session_state.knowledge_store = ingest_knowledge_store(logger)

        st.divider()
        st.subheader("Knowledge Store")
        knowledge_store = st.session_state.knowledge_store
        if knowledge_store.source_names:
            st.markdown("\n".join(f"- {source}" for source in knowledge_store.source_names))
            if knowledge_store.trusted_domains:
                st.caption(f"Trusted domains: {', '.join(knowledge_store.trusted_domains)}")
        else:
            st.info("Add knowledge files to knowledge_base so the agents can operate like experienced retail analysts.")

        st.divider()
        st.subheader("Stored Reports")
        saved_reports = sorted(INTERNAL_REPOSITORY_DIR.glob("retail_report_*.txt"), reverse=True)
        if saved_reports:
            selected_report = st.selectbox("Recent files", [report.name for report in saved_reports], index=0)
            if st.button("Load selected report"):
                report_stem = Path(selected_report).stem
                bundle_paths = storage_service.build_report_paths(report_stem)
                st.session_state.latest_report = storage_service.retrieve_report(selected_report) or ""
                st.session_state.latest_report_path = str(bundle_paths["txt"])
                st.session_state.latest_query = f"Loaded from {selected_report}"
                st.session_state.latest_pdf_path = str(bundle_paths["pdf"]) if bundle_paths["pdf"].exists() else ""
                st.session_state.latest_margin_chart_path = str(bundle_paths["margin_chart"]) if bundle_paths["margin_chart"].exists() else ""
                st.session_state.latest_alignment_chart_path = str(bundle_paths["alignment_chart"]) if bundle_paths["alignment_chart"].exists() else ""
                st.session_state.latest_summary_image_path = str(bundle_paths["summary_image"]) if bundle_paths["summary_image"].exists() else ""
                st.session_state.latest_pdf_bytes = bundle_paths["pdf"].read_bytes() if bundle_paths["pdf"].exists() else b""
                st.session_state.latest_comparison_rows = parse_comparison_table(st.session_state.latest_report)
                logger.write(f"Loaded saved report {selected_report}.")
        else:
            st.caption("No saved reports yet.")

    query = st.text_input(
        "Enter your retail research query",
        placeholder="Example: High-margin wellness trends for D2C brands in India",
    )
    run_clicked = st.button("Generate Report", type="primary")

    if run_clicked:
        if not query.strip():
            st.warning("Enter a research query to start the agents.")
        else:
            logger.reset()
            try:
                st.session_state.knowledge_store = ingest_knowledge_store(logger)
                with st.spinner("Agents are researching and drafting the report..."):
                    run_research(
                        query.strip(),
                        logger,
                        storage_service,
                        st.session_state.knowledge_store,
                    )
                st.success("Report generated successfully.")
            except Exception as error:
                logger.write(f"Run failed: {error}")
                st.error(str(error))

    if st.session_state.latest_query:
        st.subheader("Research Query")
        st.write(st.session_state.latest_query)

    if st.session_state.latest_report:
        st.subheader("Market Research Report")
        if st.session_state.latest_summary_image_path and Path(st.session_state.latest_summary_image_path).exists():
            st.image(st.session_state.latest_summary_image_path, use_container_width=True)
        if st.session_state.latest_comparison_rows:
            st.subheader("Opportunity Comparison Table")
            table_data = [
                {
                    "Product Name": row.product_name,
                    "Est. Margin": row.est_margin,
                    "Trend Alignment": row.trend_alignment,
                }
                for row in st.session_state.latest_comparison_rows
            ]
            st.table(table_data)
        else:
            st.info("The model did not return a usable comparison table in this run, so charts may be limited.")

        st.markdown(st.session_state.latest_report)
        if st.session_state.latest_comparison_rows:
            st.subheader("Opportunity Visuals")
            chart_left, chart_right = st.columns(2)
            if st.session_state.latest_margin_chart_path and Path(st.session_state.latest_margin_chart_path).exists():
                with chart_left:
                    st.image(st.session_state.latest_margin_chart_path, caption="Estimated Margin by Opportunity", use_container_width=True)
            if st.session_state.latest_alignment_chart_path and Path(st.session_state.latest_alignment_chart_path).exists():
                with chart_right:
                    st.image(st.session_state.latest_alignment_chart_path, caption="Trend Alignment Score", use_container_width=True)
        st.download_button(
            label="Download Professional Report (.txt)",
            data=st.session_state.latest_report,
            file_name=Path(st.session_state.latest_report_path).name if st.session_state.latest_report_path else "retail_report.txt",
            mime="text/plain",
        )
        if st.session_state.latest_pdf_bytes:
            st.download_button(
                label="Download Professional Report (.pdf)",
                data=st.session_state.latest_pdf_bytes,
                file_name=Path(st.session_state.latest_pdf_path).name if st.session_state.latest_pdf_path else "retail_report.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
