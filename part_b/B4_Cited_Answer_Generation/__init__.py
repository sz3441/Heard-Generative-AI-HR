"""Heard B4: privacy-aware, evidence-grounded answer generation."""

from .cited_answer import BASE_MODEL_ID, DEFAULT_REFUSAL, LocalQwenGenerator, generate_cited_answer

__all__ = ["BASE_MODEL_ID", "DEFAULT_REFUSAL", "LocalQwenGenerator", "generate_cited_answer"]
