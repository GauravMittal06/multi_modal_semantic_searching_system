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
    from .retrieval import retrieve_context, format_context_for_llm, build_citations, build_explainability
except ImportError:
    from retrieval import retrieve_context, format_context_for_llm, build_citations, build_explainability


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
            "explainability": {
                "evidence_used": [],
                "relationships_used": [],
                "modalities_used": {"text": False, "table": False, "image": False},
                "retrieval_summary": {"total_elements": 0, "page_count": 0, "type_counts": {}},
                "confidence": "Low",
            },
            "error": None,
        }

    formatted_context = format_context_for_llm(context_elements)

    doc_count = max((e.get("_doc_count", 1) for e in context_elements), default=1)

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
    4. Never invent data. Use only what is explicitly stated in the context. {"If the context contains partial information from some documents but not others, synthesize what is available and clearly note what is missing. Only say the documents do not contain sufficient information if the context contains nothing relevant at all." if doc_count > 1 else 'If context is insufficient, say: "The document does not contain sufficient information to answer this question."'}
    5. Numbers in context written as "USD" mean "$" and "pct" means "%".
    6. If two or more pieces of evidence equally and independently support different conclusions, do NOT invent a tiebreaker (such as order of mention, alphabetical order, or position in a sentence) to force a single answer. Instead, explicitly state that the evidence is split, name each supporting conclusion, and explain exactly what evidence supports each one. Only pick a single answer if the evidence itself — not your own assumption — clearly favors one over the other.
    7. If a TABLE element lacks a [TABLE COLUMNS:] header, treat the first pipe-separated row as the column header row when interpreting it.
    8. When answering questions about a specific Figure or Table, prioritize evidence from the referenced Figure/Table and its directly related paragraphs over general document discussion.
    9. If a Figure or Table contains the exact answer, answer from that evidence first before using supporting narrative text.
    10. Prefer concise answers. Do not list unrelated values, scenarios, or metrics unless they are required to answer the question.
    {"11. Multiple documents are in context. When synthesizing across them, explicitly attribute each fact to its source document by name. Never merge facts from different documents without labeling which came from where." if doc_count > 1 else ""}
CONTEXT:
{formatted_context}

QUESTION: {question}

ANSWER:"""

    # ===== STAGE_F_PROBE: confirm whether Priority 2 reached the prompt, print raw prompt =====
    _priority2_in_prompt = "Priority 2" in formatted_context
    print(f"\n[STAGE_F_PROBE] Priority 2 present in formatted_context fed to prompt? -> {_priority2_in_prompt}")
    if _priority2_in_prompt:
        _idx = formatted_context.find("Priority 2")
        _snippet_start = max(0, _idx - 50)
        _snippet_end = min(len(formatted_context), _idx + 400)
        print("[STAGE_F_PROBE] ===== formatted_context SNIPPET CONTAINING 'Priority 2' =====")
        print(formatted_context[_snippet_start:_snippet_end])
        print("[STAGE_F_PROBE] ===== END SNIPPET =====\n")
    print("[STAGE_F_PROBE] ===== RAW PROMPT SENT TO GEMINI =====")
    print(prompt)
    print("[STAGE_F_PROBE] ===== END RAW PROMPT =====\n")

    print("[STAGE_F_PROBE] ===== SSP STRING CHECK IN FINAL PROMPT =====")
    for needle in ["SSP1-1.9", "SSP5-8.5", "1.5°C", "1.6°C"]:
        print(f"[STAGE_F_PROBE] prompt contains {needle!r}? -> {needle in prompt}")
    print("[STAGE_F_PROBE] ===== END SSP STRING CHECK =====\n")

    # Step 4: Generate with Gemini
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        answer = response.text or "No answer generated."
        # ===== STAGE_F_PROBE: raw Gemini response, unmodified =====
        print("[STAGE_F_PROBE] ===== RAW GEMINI RESPONSE (response.text) =====")
        print(repr(answer))
        print("[STAGE_F_PROBE] ===== END RAW GEMINI RESPONSE =====\n")
    except Exception as e:
        fallback_citations = build_citations(context_elements, answer_text="")
        fallback_explainability = build_explainability(context_elements, fallback_citations, answer_text="")
        return {
            "answer": f"Answer generation failed: {str(e)}",
            "citations": fallback_citations,
            "context_used": context_elements,
            "explainability": fallback_explainability,
            "error": str(e),
        }

    answer_text = answer.strip()
    citations = build_citations(context_elements, answer_text=answer_text)
    explainability = build_explainability(context_elements, citations, answer_text=answer_text)

    return {
        "answer": answer_text,
        "citations": citations,
        "context_used": context_elements,
        "explainability": explainability,
        "error": None,
    }