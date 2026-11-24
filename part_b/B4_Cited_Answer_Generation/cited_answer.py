"""Evidence-grounded cited answer generation for Heard (B4)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Callable, Mapping, Optional


BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_REFUSAL = "I don't have enough evidence in the retrieved employee feedback to answer that question."
_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_MIN_QUOTE_LENGTH = 20
_GENERIC_QUERY_WORDS = {
    "employee", "employees", "staff", "worker", "workers", "people", "review",
    "reviews", "feedback", "report", "reports", "mention", "say", "unhappy",
    "concern", "concerns", "problem", "problems", "issue", "issues",
}


def _topic_terms(text: str) -> set[str]:
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stemmer = PorterStemmer()
    words = re.findall(r"[a-z]+", text.lower())
    return {
        stemmer.stem(word)
        for word in words
        if len(word) > 2 and word not in ENGLISH_STOP_WORDS
        and word not in _GENERIC_QUERY_WORDS
    }


class LocalQwenGenerator:
    """Lazy, CPU-only generator using the same base model as B2, without its ABSA adapter."""

    def __init__(self, model_id: str = BASE_MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        ).to("cpu")
        self.model.eval()

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def _default_generator() -> LocalQwenGenerator:
    """Load one model per application process rather than once per question."""
    return LocalQwenGenerator()


def _clean_results(results, min_results: int):
    """Keep only usable evidence rows and deduplicate passage IDs."""
    required = {"passage_id", "text"}
    if results is None or not required.issubset(results.columns):
        return None
    cleaned = results.copy()
    cleaned["passage_id"] = cleaned["passage_id"].fillna("").astype(str).str.strip()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    cleaned = cleaned[(cleaned["passage_id"] != "") & (cleaned["text"] != "")]
    cleaned = cleaned.drop_duplicates(subset="passage_id", keep="first")
    return cleaned if len(cleaned) >= min_results else None


def _parse_model_output(raw: str) -> tuple[str, list[str]]:
    """Accept the requested JSON format, with a conservative plain-text fallback."""
    candidate = raw.strip()
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            answer = str(payload.get("answer", "")).strip()
            citations = payload.get("citations", payload.get("citation", []))
            if not isinstance(citations, list):
                citations = []
            normalized = []
            for citation in citations:
                if isinstance(citation, str):
                    value = citation.strip()
                    if value:
                        normalized.append(value)
                    continue
                if isinstance(citation, Mapping):
                    # Small instruction models sometimes return citation objects.
                    # Prefer an exact passage_id, then recover bracketed IDs from
                    # any string value such as {"text": "[source:id:cons:0]"}.
                    exact = citation.get("passage_id")
                    if isinstance(exact, str) and exact.strip():
                        normalized.append(exact.strip())
                    for value in citation.values():
                        if isinstance(value, str):
                            normalized.extend(c.strip() for c in _CITATION_RE.findall(value))
            return answer, normalized
        except json.JSONDecodeError:
            pass
    return candidate, [c.strip() for c in _CITATION_RE.findall(candidate)]


def generate_cited_answer(
    index,
    question: str,
    *,
    search_method: str = "semantic",
    k: int = 5,
    field: Optional[str] = None,
    filters: Optional[Mapping[str, object]] = None,
    min_evidence: int = 2,
    generator: Optional[Callable[[str, str], str]] = None,
    safe_search_fn=None,
    semantic_search_fn=None,
    hybrid_search_fn=None,
) -> dict:
    """Retrieve safely, answer from evidence, validate citations, or refuse.

    Search functions may be injected for tests. In the application they default to
    A's retrieval.search implementation.
    """
    if not question or not question.strip():
        return {"answer": DEFAULT_REFUSAL, "citations": [], "refused": True, "reason": "no_evidence"}

    if safe_search_fn is None:
        from retrieval.search import hybrid_search, safe_search, semantic_search

        safe_search_fn = safe_search
        semantic_search_fn = semantic_search
        hybrid_search_fn = hybrid_search

    methods = {
        "semantic": semantic_search_fn,
        "hybrid": hybrid_search_fn,
    }
    if search_method not in methods:
        raise ValueError("search_method must be 'semantic' or 'hybrid'")

    retrieved = safe_search_fn(
        index,
        question,
        search_fn=methods[search_method],
        k=k,
        field=field,
        filters=dict(filters or {}),
    )
    if retrieved.get("suppressed", False):
        return {
            "answer": str(retrieved.get("message") or DEFAULT_REFUSAL),
            "citations": [],
            "refused": True,
            "reason": "suppressed",
        }

    evidence = _clean_results(retrieved.get("results"), min_evidence)
    if evidence is None:
        return {"answer": DEFAULT_REFUSAL, "citations": [], "refused": True, "reason": "no_evidence"}

    question_topics = _topic_terms(question)
    matched_topics = {
        row.passage_id: question_topics & _topic_terms(row.text)
        for row in evidence[["passage_id", "text"]].itertuples(index=False)
    }
    min_topics = (len(question_topics) + 1) // 2
    if len(set().union(*matched_topics.values())) < min_topics:
        return {"answer": DEFAULT_REFUSAL, "citations": [], "refused": True, "reason": "no_evidence"}

    evidence_lines = [
        f"[{row.passage_id}] {row.text}"
        for row in evidence[["passage_id", "text"]].itertuples(index=False)
    ]
    system_prompt = (
        "You answer HR questions using only the supplied employee-feedback passages. "
        "Do not use outside knowledge or infer facts not supported by the passages. "
        "Every factual claim must be supported. Return one strict JSON object with exactly "
        "two keys: answer and citations. citations must be a JSON list of strings copied "
        "character-for-character from the Allowed passage IDs supplied by the user. Never "
        "invent, shorten, reformat, or explain an ID, and never return citation objects. "
        "If the evidence is insufficient, return an empty answer and "
        "an empty citations list. Do not reveal reviewer identities or group sizes."
    )
    allowed_lines = "\n".join(f"- {pid}" for pid in evidence["passage_id"])
    user_prompt = (
        "Question:\n" + question.strip()
        + "\n\nAllowed passage IDs (citations may contain only exact strings from this list):\n"
        + allowed_lines
        + "\n\nEvidence:\n"
        + "\n".join(evidence_lines)
    )
    raw = (generator or _default_generator())(system_prompt, user_prompt)
    answer, proposed = _parse_model_output(raw)

    allowed = set(evidence["passage_id"])
    allowed_by_casefold = {}
    for passage_id in allowed:
        allowed_by_casefold.setdefault(passage_id.casefold(), []).append(passage_id)
    evidence_text = {
        row.passage_id: row.text.casefold()
        for row in evidence[["passage_id", "text"]].itertuples(index=False)
    }
    citations = []
    for proposed_id in proposed:
        if proposed_id in allowed:
            resolved = proposed_id
        else:
            matches = allowed_by_casefold.get(proposed_id.casefold(), [])
            resolved = matches[0] if len(matches) == 1 else None
            # Qwen 0.5B sometimes copies a supporting quotation instead of its
            # passage ID. Accept it only when a sufficiently long exact quote
            # occurs in exactly one of the passages from this retrieval call.
            if resolved is None and len(proposed_id.strip()) >= _MIN_QUOTE_LENGTH:
                quote = proposed_id.strip().casefold()
                quote_matches = [
                    passage_id
                    for passage_id, text in evidence_text.items()
                    if quote in text
                ]
                resolved = quote_matches[0] if len(quote_matches) == 1 else None
        if resolved and resolved not in citations:
            citations.append(resolved)
    cited_topics = set().union(*(matched_topics[pid] for pid in citations))
    if not answer or not citations or len(cited_topics) < min_topics:
        return {"answer": DEFAULT_REFUSAL, "citations": [], "refused": True, "reason": "no_evidence"}

    # Strip hallucinated bracket citations from the visible answer.
    answer = _CITATION_RE.sub(lambda m: m.group(0) if m.group(1) in allowed else "", answer)
    return {"answer": answer.strip(), "citations": citations, "refused": False, "reason": None}
