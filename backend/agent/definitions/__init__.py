"""
Agent definitions (one "recipe" per agent). Importing this package registers
every agent in ``AGENT_DEFINITIONS`` so the router can look them up by type.
"""
from agent.definitions import file_review  # noqa: F401  (registers "file_review")
from agent.definitions import engagement_buildout  # noqa: F401  (registers "engagement_buildout")
from agent.definitions import analytical_review  # noqa: F401  (registers "analytical_review")
from agent.definitions import substantive_testing  # noqa: F401  (registers "substantive_testing")
from agent.definitions import risk_assessment  # noqa: F401  (registers "risk_assessment")
from agent.definitions import review_notes  # noqa: F401  (registers "review_notes")
from agent.definitions import document_intelligence  # noqa: F401  (registers "document_intelligence")
from agent.definitions import procedure_buildout  # noqa: F401  (registers "procedure_buildout")

__all__ = ["file_review", "engagement_buildout", "analytical_review", "substantive_testing", "risk_assessment", "review_notes", "document_intelligence", "procedure_buildout"]
