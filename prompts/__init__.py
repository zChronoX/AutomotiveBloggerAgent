"""Modulo prompt: tutti i template centralizzati."""

from .system_prompts import blogger_system_prompt, default_background, default_editorial_guidelines
from .agent_prompts import PLANNING_PROMPT, RESEARCH_KICKOFF, DRAFT_PROMPT

__all__ = [
    "blogger_system_prompt",
    "default_background",
    "default_editorial_guidelines",
    "PLANNING_PROMPT",
    "RESEARCH_KICKOFF",
    "DRAFT_PROMPT",
]
