# src/answer.py
"""
Answer Generation — Module 6
Sends combined multi-modal context to Gemini and returns a grounded answer with citations.
"""

from typing import List, Dict, Any

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import GEMINI_KEY, GEMINI_MODEL

try:
    from .retrieval import retrieve_context, format_context_for_llm, build_citations
except ImportError:
    from retrieval import retrieve_context, format_context_for_llm, build_citations


def generate_answer(
    question: str,
    source_document: str = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Full QA pipeline:
    1. Retrieve multi-modal context (paragraphs + tables + images)
    2. Format context for LLM
    3. Generate grounded answer with Gemini
    4. Return answer + citations + context used

    Returns dict with keys: answer, citations, context_used, error
    """
    # Step 1–2: Retrieve and format context
    context_elements = retrieve_context(
        question=question,
        source_document=source_document,
        top_k=top_k,
        expand_related=True,
    )

    if not context_elements:
        return {
            "answer": "I could not find relevant information in the document to answer this question.",
            "citations": [],
            "context_used": [],
            "error": None,
        }

    formatted_context = format_context_for_llm(context_elements)
    citations = build_citations(context_elements)

    # Step 3: Build prompt
    prompt = f"""You are an expert document analyst. Answer questions using ONLY the provided document context below.

Context element types:
- PARAGRAPH: body text
- TABLE: structured data (numbers, metrics, comparisons) — treat pipe-separated rows as tabular data
- IMAGE: AI-generated semantic description of a chart or figure — treat this as factual visual evidence
- HEADING: section label only, not a source of facts

Rules:
1. Cross-reference paragraphs WITH tables AND images to form complete answers.
2. When a TABLE contains relevant numbers, quote them explicitly.
3. When an IMAGE description contains trends or insights, cite them as visual evidence.
4. Never invent data. If context is insufficient, say: "The document does not contain sufficient information to answer this question."
5. Numbers in context written as "USD" mean "$" and "pct" means "%".

CONTEXT:
{formatted_context}

QUESTION: {question}

ANSWER:"""

    # Step 4: Generate with Gemini
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        answer = response.text or "No answer generated."
    except Exception as e:
        return {
            "answer": f"Answer generation failed: {str(e)}",
            "citations": citations,
            "context_used": context_elements,
            "error": str(e),
        }

    return {
        "answer": answer.strip(),
        "citations": citations,
        "context_used": context_elements,
        "error": None,
    }