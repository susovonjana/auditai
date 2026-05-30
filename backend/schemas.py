"""
Pydantic request/response schemas for the AuditAI API.
"""
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


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
    status: str
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
class SessionStartResponse(BaseModel):
    session_token: str
    started_at: datetime


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


# =========================================================================
# Q&A
# =========================================================================
class AskRequest(BaseModel):
    session_token: str
    question: str = Field(..., min_length=1, max_length=4000)
    language: str = Field("en", pattern="^(en|ar)$")  # response language


class AskResponse(BaseModel):
    answer: str
    history_id: UUID
    was_answered: bool
    response_time_ms: int
    documents_referenced: List[str] = []


class FeedbackRequest(BaseModel):
    history_id: UUID
    feedback: str = Field(..., pattern="^(helpful|not_helpful)$")


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
