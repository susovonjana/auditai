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
