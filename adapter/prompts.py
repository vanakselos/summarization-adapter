"""Prompt templates for summarization tasks."""

from __future__ import annotations


def _build_focus_block(query: str | None) -> str:
    query_block = ""
    if query:
        query_block = f"\n### Focus:\n{query.strip()}\n"
    return query_block


def build_chunk_summary_prompt(
    section_text: str,
    max_words: int = 120,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Summarize the key information in this section of the document. "
        "Focus on the most important facts, findings, actions, or outcomes. "
        "Avoid generic filler and repetition. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Section:\n"
        f"{section_text.strip()}\n\n"
        "### Section Summary:\n"
    )


def build_merge_summary_prompt(
    section_summaries: str,
    max_words: int = 220,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Combine the section summaries into coherent notes for a final summary. "
        "Remove redundancy and keep only the most important points. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Section Summaries:\n"
        f"{section_summaries.strip()}\n\n"
        "### Consolidated Notes:\n"
    )


def build_xsum_chunk_prompt(
    section_text: str,
    max_words: int = 60,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Summarize the key facts in this section of the news article. "
        "Focus on the main event, the main actor, and the outcome. "
        "Keep only details needed for a final one-sentence news summary. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Article Section:\n"
        f"{section_text.strip()}\n\n"
        "### News Notes:\n"
    )


def build_xsum_merge_prompt(
    section_summaries: str,
    max_words: int = 80,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Combine the section notes into concise notes for a final one-sentence news summary. "
        "Keep only the central event, the main actor, and the outcome. "
        "Remove background details and repetition. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Section Notes:\n"
        f"{section_summaries.strip()}\n\n"
        "### Final News Notes:\n"
    )


def build_govreport_chunk_prompt(
    section_text: str,
    max_words: int = 120,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Summarize the key information in this section of the government report. "
        "Focus on major findings, actions, recommendations, and impacts. "
        "Use clear factual language and avoid repetition. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Report Section:\n"
        f"{section_text.strip()}\n\n"
        "### Section Notes:\n"
    )


def build_govreport_merge_prompt(
    section_summaries: str,
    max_words: int = 220,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Combine the section notes into coherent notes for an executive summary of the government report. "
        "Keep the most important findings, actions, recommendations, and impacts. "
        "Remove redundancy and keep the flow clear. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Section Notes:\n"
        f"{section_summaries.strip()}\n\n"
        "### Executive Summary Notes:\n"
    )


def build_arxiv_intro_prompt(
    section_text: str,
    max_words: int = 100,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "You are a scientific research assistant. Summarize this opening section of a paper. "
        "Focus on: 1. the research objective, 2. the problem setting, 3. the main approach introduced. "
        "Avoid repetition and avoid unnecessary mathematical notation. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Paper Opening:\n"
        f"{section_text.strip()}\n\n"
        "### Introduction Context:\n"
    )


def build_arxiv_findings_prompt(
    section_text: str,
    max_words: int = 80,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "You are a scientific research assistant. Extract concise research notes from this middle section of a paper. "
        "Capture concrete methods, named entities, datasets, evidence, and the most important quantitative or qualitative findings. "
        "Do not summarize the whole section and avoid generic phrases such as 'this paper presents'. "
        "Avoid repetition and avoid unnecessary mathematical notation. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Paper Section:\n"
        f"{section_text.strip()}\n\n"
        "### Key Findings:\n"
    )


def build_arxiv_conclusion_prompt(
    section_text: str,
    max_words: int = 100,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "You are a scientific research assistant. Summarize this concluding section of a paper. "
        "Focus on the final conclusions, strongest results, implications, and clearly stated limitations or future work. "
        "Avoid repetition and avoid unnecessary mathematical notation. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Paper Conclusion:\n"
        f"{section_text.strip()}\n\n"
        "### Conclusion Notes:\n"
    )


def build_arxiv_final_abstract_prompt(
    notes: str,
    max_words: int = 150,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "You are a scientific research assistant. Write a concise summary of the paper using the notes below. "
        "Focus on: 1. the research objective, 2. the main method, 3. the most important result. "
        "Remove repetition, keep the flow coherent, and avoid unnecessary mathematical notation. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Research Notes:\n"
        f"{notes.strip()}\n\n"
        "### Abstract:\n"
    )


def build_arxiv_paper_summary_prompt(
    document: str,
    max_words: int = 150,
    query: str | None = None,
) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "You are a scientific research assistant. Summarize the following paper. "
        "Focus on: 1. the research objective, 2. the main method, 3. the most important result. "
        "Avoid repetition and avoid unnecessary mathematical notation. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Document:\n"
        f"{document.strip()}\n\n"
        "### Abstract:\n"
    )


def build_final_abstract_prompt(document: str, max_words: int = 150, query: str | None = None) -> str:
    """Backward-compatible wrapper for the general summary prompt."""
    return build_general_summary_prompt(document, max_words=max_words, query=query)


def build_general_summary_prompt(document: str, max_words: int = 150, query: str | None = None) -> str:
    query_block = _build_focus_block(query)

    return (
        "### Instruction:\n"
        "Write a concise summary of the following document. "
        "Focus on the central issue, the key findings or events, and the most important outcomes. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Document:\n"
        f"{document.strip()}\n\n"
        "### Summary:\n"
    )


def build_xsum_prompt(document: str, max_words: int = 35, query: str | None = None) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Write a single-sentence news summary of the following article. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Article:\n"
        f"{document.strip()}\n\n"
        "### Summary:\n"
    )


def build_govreport_prompt(document: str, max_words: int = 180, query: str | None = None) -> str:
    query_block = _build_focus_block(query)
    return (
        "### Instruction:\n"
        "Write a concise executive summary of the following government report. "
        f"Use at most {max_words} words.\n"
        f"{query_block}"
        "\n### Report:\n"
        f"{document.strip()}\n\n"
        "### Executive Summary:\n"
    )


def build_summary_prompt(document: str, max_words: int = 150, query: str | None = None) -> str:
    """Backward-compatible alias used by older training/inference code paths."""
    return build_general_summary_prompt(document, max_words=max_words, query=query)


def strip_prompt_from_generation(generated_text: str, prompt: str) -> str:
    """Remove echoed prompt from model output."""
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    return generated_text.strip()
