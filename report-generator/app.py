"""
Multi-Agent Report Generator

Workflow:
Research Agent
    ↓
Summary Agent
    ↓
Writing Agent
    ↓
Review Agent
"""

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

# This loads the API keys stored inside the .env file.
load_dotenv()


# --------------------------------------------------
# 2. Shared LangGraph state
# --------------------------------------------------

class ReportState(TypedDict):
    """
    This is the shared state that moves through the graph.

    Every agent reads information from this state and
    saves its result back into the state.
    """

    topic: str
    research_notes: str
    summary: str
    draft_report: str
    final_report: str


# --------------------------------------------------
# 3. Check API keys
# --------------------------------------------------

def validate_api_keys() -> None:
    """
    Check that the required API keys exist before
    starting the application.
    """

    missing_keys = []

    if not os.getenv("GROQ_API_KEY"):
        missing_keys.append("GROQ_API_KEY")

    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        keys = ", ".join(missing_keys)

        raise RuntimeError(
            f"Missing API key(s): {keys}. "
            "Please add them to your .env file."
        )


# --------------------------------------------------
# 4. Create the AI model and search tool
# --------------------------------------------------

def create_llm() -> ChatGroq:
    """
    Create the Groq language model used by all agents.
    """

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )


def create_search_tool() -> TavilySearch:
    return TavilySearch(max_results=5)


# These variables will be initialized inside main().
llm: ChatGroq
search_tool: TavilySearch


# --------------------------------------------------
# 5. Research Agent
# --------------------------------------------------

def research_agent(state: ReportState) -> ReportState:
    """
    Search for information and create detailed research notes.
    """

    topic = state["topic"]

    print("1. Research Agent is collecting information...")

    search_results = search_tool.invoke(topic)

    prompt = f"""
You are an expert Research Agent in a multi-agent AI system.

Your responsibility is ONLY to collect and organize information.

You are NOT responsible for summarizing, writing a report,
or reviewing content.

Research topic:

{topic}

Search results:

{search_results}

Create comprehensive research notes that will be passed
to another AI agent.

Organize the research notes using these sections:

- Main Ideas
- Important Facts
- Statistics
- Recent Developments
- Applications
- Benefits
- Challenges
- Future Directions
- Sources

Requirements:

- Collect as much relevant information as possible.
- Preserve important facts and technical details.
- Include statistics exactly as reported when available.
- Include recent developments and trends.
- Include applications, benefits, and challenges.
- Mention source titles or URLs when available.
- Use clear headings and bullet points.

Do NOT:

- Summarize the information.
- Write a finished report.
- Remove useful details.
- Add unsupported information.

Your output should be professional research notes prepared
for the next AI agent.
"""

    response = llm.invoke(prompt)

    state["research_notes"] = response.content

    return state


# --------------------------------------------------
# 6. Summary Agent
# --------------------------------------------------

def summary_agent(state: ReportState) -> ReportState:
    """
    Summarize the research notes without adding information.
    """

    print("2. Summary Agent is summarizing the research...")

    research_notes = state["research_notes"]

    prompt = f"""
You are an expert Summary Agent in a multi-agent AI system.

Your responsibility is ONLY to summarize the research notes.

You are NOT responsible for conducting research,
writing a complete report, or reviewing content.

Research notes:

{research_notes}

Create a clear and concise summary.

Requirements:

- Preserve the main ideas.
- Preserve important facts.
- Preserve important statistics.
- Preserve major trends.
- Preserve important benefits and challenges.
- Remove unnecessary repetition.
- Use clear and professional language.
- Organize the summary into short paragraphs or bullet points.

Do NOT:

- Add new information.
- Change the meaning of the research.
- Write a complete business report.
- Give personal opinions.
- Add unsupported recommendations.

Your output will be passed to the Writing Agent.
"""

    response = llm.invoke(prompt)

    state["summary"] = response.content

    return state


# --------------------------------------------------
# 7. Writing Agent
# --------------------------------------------------

def writing_agent(state: ReportState) -> ReportState:
    """
    Convert the summary into a professional business report.
    """

    print("3. Writing Agent is creating the report...")

    topic = state["topic"]
    summary = state["summary"]

    prompt = f"""
You are a professional Business Report Writer
in a multi-agent AI system.

Your responsibility is ONLY to write a professional report
using the summary provided.

You are NOT responsible for conducting new research or
reviewing the report.

Report topic:

{topic}

Summary:

{summary}

Write a professional report using this structure:

# Executive Summary

Provide a concise overview of the topic and the major findings.

# Introduction

Introduce the topic and explain why it is important.

# Key Findings

Present the most important findings clearly.
Use bullet points where appropriate.

# Benefits

Explain the major benefits and opportunities.

# Challenges

Discuss the important challenges, risks, and limitations.

# Future Outlook

Explain expected developments and future opportunities.

# Conclusion

Provide a clear and professional closing summary.

Requirements:

- Use a formal and professional tone.
- Use clear business-style English.
- Use clear headings.
- Create smooth transitions between sections.
- Use only information from the provided summary.
- Preserve important facts and statistics.

Do NOT:

- Add unsupported claims.
- Invent facts or statistics.
- Repeat the same information unnecessarily.
- Review or critique the report.
"""

    response = llm.invoke(prompt)

    state["draft_report"] = response.content

    return state


# --------------------------------------------------
# 8. Review Agent
# --------------------------------------------------

def review_agent(state: ReportState) -> ReportState:
    """
    Review and improve the draft report.
    """

    print("4. Review Agent is reviewing the report...")

    draft_report = state["draft_report"]

    prompt = f"""
You are a professional Report Review Agent
in a multi-agent AI system.

Your responsibility is ONLY to review and improve
the draft report.

You are NOT responsible for conducting research,
summarizing information, or rewriting the entire report
from scratch.

Draft report:

{draft_report}

Review and improve the report.

Requirements:

- Correct grammar and spelling.
- Improve clarity and readability.
- Improve sentence flow.
- Improve transitions between sections.
- Remove unnecessary repetition.
- Maintain a professional business tone.
- Preserve all facts and statistics.
- Preserve the existing report structure.
- Ensure every section is clear and polished.

Do NOT:

- Add unsupported information.
- Remove important facts.
- Change the meaning of the report.
- Change the report structure.
- Add personal opinions.

Return ONLY the improved final report.
"""

    response = llm.invoke(prompt)

    state["final_report"] = response.content

    return state


# --------------------------------------------------
# 9. Build the LangGraph workflow
# --------------------------------------------------

def build_workflow():
    """
    Build the graph that acts as the Report Manager.

    The graph coordinates the agents in this order:

    Research -> Summary -> Writing -> Review
    """

    graph = StateGraph(ReportState)

    # Add the four agents as graph nodes.
    graph.add_node("research", research_agent)
    graph.add_node("summary", summary_agent)
    graph.add_node("writing", writing_agent)
    graph.add_node("review", review_agent)

    # Connect the agents in the correct order.
    graph.add_edge(START, "research")
    graph.add_edge("research", "summary")
    graph.add_edge("summary", "writing")
    graph.add_edge("writing", "review")
    graph.add_edge("review", END)

    # Compile the graph into a runnable application.
    return graph.compile()


# --------------------------------------------------
# 10. Save the final report
# --------------------------------------------------

def save_report(report: str) -> Path:
    """
    Save the generated report as a Markdown file.

    By default, it will be saved inside:

    reports/final_report.md
    """

    reports_directory = Path(
        os.getenv("REPORTS_DIR", "reports")
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = reports_directory / "final_report.md"

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return report_path


# --------------------------------------------------
# 11. Get the topic
# --------------------------------------------------

def get_topic() -> str:
    """
    Get the report topic.

    The user can provide the topic in two ways:

    1. Enter it after the application starts.
    2. Pass it directly through the terminal.
    """

    # Example:
    # python app.py "Artificial Intelligence in Healthcare"
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    # Example:
    # python app.py
    # Then the application asks for the topic.
    return input("Enter a report topic: ").strip()


# --------------------------------------------------
# 12. Main application
# --------------------------------------------------

def main() -> None:
    """
    Start and run the complete application.
    """

    global llm
    global search_tool

    print()
    print("=" * 60)
    print("MULTI-AGENT REPORT GENERATOR")
    print("=" * 60)

    try:
        # Check that the API keys exist.
        validate_api_keys()

        # Create the model and search tool.
        llm = create_llm()
        search_tool = create_search_tool()

        # Ask the user for a topic.
        topic = get_topic()

        if not topic:
            raise ValueError(
                "The report topic cannot be empty."
            )

        # This is the initial state given to LangGraph.
        initial_state: ReportState = {
            "topic": topic,
            "research_notes": "",
            "summary": "",
            "draft_report": "",
            "final_report": "",
        }

        print()
        print(f"Topic: {topic}")
        print()
        print("Starting the multi-agent workflow...")
        print()

        # Build the graph.
        workflow = build_workflow()

        # Run the graph.
        result = workflow.invoke(initial_state)

        # Get the final result created by the Review Agent.
        final_report = result["final_report"]

        # Save the report.
        report_path = save_report(final_report)

        print()
        print("=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        print()
        print(final_report)
        print()
        print("=" * 60)
        print(f"Report saved to: {report_path.resolve()}")
        print("=" * 60)

    except (RuntimeError, ValueError) as error:
        print()
        print(f"Configuration error: {error}")
        raise SystemExit(1) from error

    except Exception as error:
        print()
        print(f"The report could not be generated: {error}")
        raise SystemExit(1) from error


# --------------------------------------------------
# 13. Application entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()