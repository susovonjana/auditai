"""
AuditAI supervised agent runtime.

A small plan -> execute -> checkpoint engine that drives multi-step audit
assistants (file review, engagement build-out, ...). It reuses the existing
copilot read tools, the Bedrock structured-output helpers, and the copilot
grant for auth. Every step and approval is persisted to ``agent_runs`` /
``agent_steps`` so a run is resumable and fully auditable.

Nothing in this package is autonomous: every step that WRITES to an audit file
pauses for human approval (ISA 220 / ISQM 1), and no financial figure is ever
computed by the LLM -- numbers come from tools or deterministic code.
"""
