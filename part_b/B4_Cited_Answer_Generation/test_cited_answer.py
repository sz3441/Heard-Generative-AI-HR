import pandas as pd

try:
    # Normal pytest collection: this directory is a Python package.
    from .cited_answer import DEFAULT_REFUSAL, generate_cited_answer
except ImportError:
    # Also support running this file directly from the B4 directory.
    from cited_answer import DEFAULT_REFUSAL, generate_cited_answer


PASSAGES = pd.DataFrame(
    [
        {"passage_id": "p:1", "text": "Promotion criteria are unclear."},
        {"passage_id": "p:2", "text": "Career progression feels slow."},
    ]
)


def _safe(payload):
    def call(*args, **kwargs):
        return payload
    return call


def test_suppression_uses_safe_search_message():
    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": True, "results": PASSAGES, "message": "Privacy refusal."}
        ), semantic_search_fn=object(), hybrid_search_fn=object()
    )
    assert out == {"answer": "Privacy refusal.", "citations": [], "refused": True}


def test_invalid_citations_are_filtered():
    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": False, "results": PASSAGES, "message": ""}
        ), semantic_search_fn=object(), hybrid_search_fn=object(),
        generator=lambda *_: '{"answer":"Promotion is unclear.","citations":["p:1","fake"]}',
    )
    assert out["citations"] == ["p:1"]
    assert not out["refused"]


def test_no_valid_citation_refuses():
    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": False, "results": PASSAGES, "message": ""}
        ), semantic_search_fn=object(), hybrid_search_fn=object(),
        generator=lambda *_: '{"answer":"Unsupported.","citations":["fake"]}',
    )
    assert out == {"answer": DEFAULT_REFUSAL, "citations": [], "refused": True}


def test_qwen_style_citation_objects_are_normalized():
    raw = (
        '{"answer":"Promotion criteria are unclear.","citations":['
        '{"id":"p","text":"[p:1]"},{"passage_id":"p:2"}]}'
    )
    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": False, "results": PASSAGES, "message": ""}
        ), semantic_search_fn=object(), hybrid_search_fn=object(),
        generator=lambda *_: raw,
    )
    assert out["citations"] == ["p:1", "p:2"]
    assert not out["refused"]


def test_qwen_singular_citation_key_is_accepted():
    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": False, "results": PASSAGES, "message": ""}
        ), semantic_search_fn=object(), hybrid_search_fn=object(),
        generator=lambda *_: '{"answer":"Supported.","citation":["p:1"]}',
    )
    assert out["citations"] == ["p:1"]
    assert not out["refused"]


def test_citation_case_is_restored_only_from_allowed_ids():
    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": False, "results": PASSAGES, "message": ""}
        ), semantic_search_fn=object(), hybrid_search_fn=object(),
        generator=lambda *_: '{"answer":"Supported.","citation":["P:1"]}',
    )
    assert out["citations"] == ["p:1"]
    assert not out["refused"]


def test_unique_exact_quote_is_mapped_to_passage_id():
    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": False, "results": PASSAGES, "message": ""}
        ), semantic_search_fn=object(), hybrid_search_fn=object(),
        generator=lambda *_: (
            '{"answer":"Supported.",'
            '"citation":["Promotion criteria are unclear."]}'
        ),
    )
    assert out["citations"] == ["p:1"]
    assert not out["refused"]


def test_short_or_nonmatching_quote_is_rejected():
    for citation in ("Promotion", "A claim not found in any retrieved passage"):
        out = generate_cited_answer(
            object(), "why?", safe_search_fn=_safe(
                {"suppressed": False, "results": PASSAGES, "message": ""}
            ), semantic_search_fn=object(), hybrid_search_fn=object(),
            generator=lambda *_, value=citation: (
                '{"answer":"Unsupported.","citation":' + repr([value]).replace("'", '"') + '}'
            ),
        )
        assert out == {"answer": DEFAULT_REFUSAL, "citations": [], "refused": True}


def test_insufficient_evidence_refuses_without_calling_generator():
    called = False

    def generator(*args):
        nonlocal called
        called = True
        return "should not run"

    out = generate_cited_answer(
        object(), "why?", safe_search_fn=_safe(
            {"suppressed": False, "results": PASSAGES.head(1), "message": ""}
        ), semantic_search_fn=object(), hybrid_search_fn=object(), generator=generator,
    )
    assert out == {"answer": DEFAULT_REFUSAL, "citations": [], "refused": True}
    assert not called


def test_search_method_is_passed_to_safe_search():
    semantic = object()
    hybrid = object()
    observed = {}

    def safe(*args, **kwargs):
        observed.update(kwargs)
        return {"suppressed": False, "results": PASSAGES, "message": ""}

    generate_cited_answer(
        object(), "why?", search_method="hybrid", safe_search_fn=safe,
        semantic_search_fn=semantic, hybrid_search_fn=hybrid,
        generator=lambda *_: '{"answer":"Supported.","citations":["p:1"]}',
    )
    assert observed["search_fn"] is hybrid
