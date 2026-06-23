"""
Pydantic request/response schemas for the AuditAI API.
"""
from datetime import datetime, date
from typing import Optional, List, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator


# =========================================================================
# Generic
# =========================================================================
class Message(BaseModel):
    message: str


# =========================================================================
# Admin auth
# =========================================================================
class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int
    username: str
    role: str


# =========================================================================
# Knowledge base status
# =========================================================================
class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    last_uploaded_filename: Optional[str] = None
    last_uploaded_at: Optional[datetime] = None
    health_status: str  # "green" | "yellow" | "red"


# =========================================================================
# Document
# =========================================================================
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    file_type: str
    category: Optional[str] = None
    status: str                      # queued | parsing | embedding | active | error
    error_message: Optional[str] = None
    total_chunks: int
    file_size_bytes: int
    uploaded_by: str
    uploaded_at: datetime
    updated_at: datetime


class DocumentUpdateRequest(BaseModel):
    category: Optional[str] = Field(None, max_length=128)


class DocumentUploadResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    total_chunks: int
    message: str


# =========================================================================
# Sessions
# =========================================================================
class SessionStartRequest(BaseModel):
    """Optional payload on POST /session/start. Callers (e.g. the 1audit
    embed) include identity so the session row carries user_id / org_id."""
    user_identifier: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    @field_validator("user_id", "organization_id", mode="before")
    @classmethod
    def _stringify_ids(cls, v):
        if v is None or v == "":
            return None
        return str(v)


class SessionStartResponse(BaseModel):
    session_token: str
    started_at: datetime


class UserUsageResponse(BaseModel):
    """Per-user (or per-session) token usage rollup for the day so far (UTC)."""
    total_tokens_today: int


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_token: str
    user_identifier: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    started_at: datetime
    last_active_at: datetime
    ended_at: Optional[datetime] = None
    total_questions: int
    user_id: Optional[str] = None
    organization_id: Optional[str] = None


# =========================================================================
# Q&A
# =========================================================================
class AskRequest(BaseModel):
    session_token: str
    question: str = Field(..., min_length=1, max_length=4000)
    language: str = Field("en", pattern="^(en|ar)$")  # response language
    # Optional caller identity (e.g. 1audit embed). Falls through to NULL
    # in search_history when absent. Accepts int or str — callers often hold
    # these as numeric IDs in their redux store.
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    @field_validator("user_id", "organization_id", mode="before")
    @classmethod
    def _stringify_ids(cls, v):
        # Pydantic v2 won't coerce int → str by default; do it here so the
        # frontend can send raw numeric IDs without serializing them first.
        if v is None or v == "":
            return None
        return str(v)


class Source(BaseModel):
    """One traceable reference contributing to an answer."""
    document: str
    page: Optional[int] = None
    section: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    history_id: UUID
    was_answered: bool
    response_time_ms: int
    documents_referenced: List[str] = []  # kept for backwards compat; prefer `sources`
    sources: List[Source] = []
    confidence: float = 0.0
    # LLM token usage (0 when no Gemini call was made).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class FeedbackRequest(BaseModel):
    history_id: UUID
    feedback: str = Field(..., pattern="^(helpful|not_helpful)$")


# =========================================================================
# Copilot — procedure generation (ticket A-1)
# =========================================================================
class ProcedureRiskItem(BaseModel):
    """One linked risk the procedure should respond to."""
    title: Optional[str] = Field(None, max_length=1000)
    description: Optional[str] = Field(None, max_length=8000)
    assessment_level: Optional[str] = Field(None, max_length=64)


class ProcedureRequest(BaseModel):
    session_token: str
    section_title: Optional[str] = Field(None, max_length=512)
    risks: List[ProcedureRiskItem] = Field(default_factory=list, max_length=20)
    assertions: List[str] = Field(default_factory=list, max_length=40)
    client_sector: Optional[str] = Field(None, max_length=256)
    audit_area: Optional[str] = Field(None, max_length=256)
    language: str = Field("en", pattern="^(en|ar)$")
    # Optional caller identity (1audit embed). Accepts int or str.
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    @field_validator("user_id", "organization_id", mode="before")
    @classmethod
    def _stringify_ids(cls, v):
        if v is None or v == "":
            return None
        return str(v)


# =========================================================================
# Copilot — procedure "house style" memory (ticket A-1b)
# =========================================================================
class ProcedureConfirmRequest(BaseModel):
    """Sent when an auditor SAVES an AI-assisted procedure, so its text becomes a
    future few-shot example for this organization (proc_memory). Org-scoped."""
    session_token: str
    section_title: Optional[str] = Field(None, max_length=512)
    audit_area: Optional[str] = Field(None, max_length=256)
    client_sector: Optional[str] = Field(None, max_length=256)
    risks: List[ProcedureRiskItem] = Field(default_factory=list, max_length=20)
    assertions: List[str] = Field(default_factory=list, max_length=40)
    procedure_html: str = Field(..., min_length=1, max_length=20000)
    language: str = Field("en", pattern="^(en|ar)$")
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    @field_validator("user_id", "organization_id", mode="before")
    @classmethod
    def _stringify_confirm_ids(cls, v):
        if v is None or v == "":
            return None
        return str(v)


# =========================================================================
# Copilot — AI draft findings (ticket A-6)
# =========================================================================
class ProcedureFindingsRequest(BaseModel):
    session_token: str
    audit_file_id: int
    copilot_grant: str
    # The procedure text the auditor is responding to (HTML or plain).
    procedure: Optional[str] = Field(None, max_length=20000)
    # Optional linkage to narrow which account's results to read.
    account: Optional[str] = Field(None, max_length=256)
    coa_original_id: Optional[int] = None
    assertions: List[str] = Field(default_factory=list, max_length=40)
    language: str = Field("en", pattern="^(en|ar)$")
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    @field_validator("user_id", "organization_id", mode="before")
    @classmethod
    def _stringify_findings_ids(cls, v):
        if v is None or v == "":
            return None
        return str(v)


# =========================================================================
# Copilot — file-grounded chat (B-3)
# =========================================================================
class CopilotChatRequest(BaseModel):
    session_token: str
    audit_file_id: int
    copilot_grant: str
    question: str = Field(..., min_length=1, max_length=4000)
    language: str = Field("en", pattern="^(en|ar)$")
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    @field_validator("user_id", "organization_id", mode="before")
    @classmethod
    def _stringify_chat_ids(cls, v):
        if v is None or v == "":
            return None
        return str(v)


# =========================================================================
# Copilot — TB auto-mapping (ticket C-3)
# =========================================================================
class TbAccountIn(BaseModel):
    tb_account_id: int
    account_name: Optional[str] = Field(None, max_length=512)
    account_name_sl: Optional[str] = Field(None, max_length=512)
    account_code: Optional[str] = Field(None, max_length=64)
    cy_amount: Optional[float] = None


class CoaCandidateIn(BaseModel):
    coa_original_id: int
    label: Optional[str] = Field(None, max_length=512)
    group: Optional[str] = Field(None, max_length=256)
    type: Optional[str] = Field(None, max_length=256)


class PriorMappingIn(BaseModel):
    account_name: Optional[str] = Field(None, max_length=512)
    account_name_sl: Optional[str] = Field(None, max_length=512)
    account_code: Optional[str] = Field(None, max_length=64)
    coa_original_id: int
    coa_label: Optional[str] = Field(None, max_length=512)


class TbMappingRequest(BaseModel):
    """Suggest a COA per unmapped TB account. The 1audit-be bridge gathers the
    unmapped accounts, the file's candidate COA set, and (when available) this
    client's prior-year confirmed mappings, and posts them here."""
    # Optional: this is a trusted be->auditai call (be already authorized via
    # route permissions). Forwarded for rate-limit scoping when the FE has a
    # copilot session open; otherwise the call still proceeds.
    session_token: Optional[str] = None
    audit_file_id: Optional[int] = None
    copilot_grant: Optional[str] = None
    organization_id: Optional[str] = None
    client_sector: Optional[str] = Field(None, max_length=256)
    prime_org_id: Optional[str] = None
    accounts: List[TbAccountIn] = Field(..., min_length=1, max_length=3000)
    coa: List[CoaCandidateIn] = Field(default_factory=list, max_length=8000)
    prior_mappings: List[PriorMappingIn] = Field(default_factory=list, max_length=8000)
    language: str = Field("en", pattern="^(en|ar)$")
    # Tier-3 LLM tail is OFF by default — Tiers 1 & 2 are fully deterministic and
    # need no Gemini quota. The bridge flips this on when quota is available.
    use_llm_tail: bool = False
    user_id: Optional[str] = None

    @field_validator("user_id", "organization_id", "prime_org_id", mode="before")
    @classmethod
    def _stringify_tbmap_ids(cls, v):
        if v is None or v == "":
            return None
        return str(v)


class TbMappingFeedbackItem(BaseModel):
    account_name: Optional[str] = Field(None, max_length=512)
    account_name_sl: Optional[str] = Field(None, max_length=512)
    account_code: Optional[str] = Field(None, max_length=64)
    coa_original_id: int
    coa_label: Optional[str] = Field(None, max_length=512)


class TbMappingFeedbackRequest(BaseModel):
    """A confirmed TB mapping (AI-accepted OR manual) posted by 1audit-be so it is
    appended to tb_mapping_memory and becomes a Tier-1 hit next time (ticket C-6).
    Best-effort: the route never raises to the caller."""
    session_token: Optional[str] = None
    organization_id: Optional[str] = None
    client_sector: Optional[str] = Field(None, max_length=256)
    confirmed_by: Optional[str] = None
    items: List[TbMappingFeedbackItem] = Field(..., min_length=1, max_length=2000)

    @field_validator("organization_id", "confirmed_by", mode="before")
    @classmethod
    def _stringify_fb_ids(cls, v):
        if v is None or v == "":
            return None
        return str(v)


# =========================================================================
# Translation
# =========================================================================
class TranslateRequest(BaseModel):
    session_token: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=10000)
    target_language: str = Field(..., pattern="^(en|ar)$")


class TranslateResponse(BaseModel):
    translated_text: str
    target_language: str
    response_time_ms: int


# =========================================================================
# Search history
# =========================================================================
class SearchHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    question: str
    ai_answer: str
    chunks_used: Any
    documents_referenced: Any
    similarity_scores: Any
    response_time_ms: int
    was_answered: bool
    user_feedback: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    asked_at: datetime


class SearchHistoryPage(BaseModel):
    items: List[SearchHistoryItem]
    total: int
    page: int
    page_size: int


# =========================================================================
# Analytics
# =========================================================================
class AnalyticsSummary(BaseModel):
    total_questions_today: int
    total_questions_week: int
    total_questions_all_time: int
    answer_success_rate: float  # 0..1
    avg_response_time_ms: float
    unique_sessions: int
    most_active_hour: Optional[int] = None  # 0..23


class TopTopic(BaseModel):
    keyword: str
    count: int


class TopTopicsResponse(BaseModel):
    topics: List[TopTopic]


# Token usage rollup by (day, user_id, organization_id)
class UsageByUserRow(BaseModel):
    day: date
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    questions: int
    total_tokens: int


class UsageByUserPage(BaseModel):
    items: List[UsageByUserRow]
    total: int
    page: int
    page_size: int
