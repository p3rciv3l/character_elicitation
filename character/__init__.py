"""
Character elicitation module.

This module provides tools for eliciting personality trait preferences from models
through pairwise comparisons and LLM-as-judge evaluation.
"""

from character.utils import traits, gen_args, load_model_and_tokenizer
from character.elicitation import preferences_vllm
from character.judgements import judge, parse_answer

__all__ = [
    "traits",
    "gen_args", 
    "load_model_and_tokenizer",
    "preferences_vllm",
    "judge",
    "parse_answer",
]
