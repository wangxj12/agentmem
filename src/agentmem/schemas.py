"""Pydantic request and response models for the add and search endpoints."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MessagePayload(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    timestamp: int | None = None
    speaker: str | None = Field(default=None, min_length=1)


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    messages: list[MessagePayload] = Field(min_length=1)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str
    complete: bool = False
    channels: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(gt=0, le=100)


class ChunkMetadataRequest(BaseModel):
    user_id: str = Field(min_length=1)
    ids: list[str] = Field(max_length=100)


class CommonHit(BaseModel):
    id: str
    content: str
    score: float
    created_at: str


class ChunkDetails(BaseModel):
    speaker: str | None = None
    role: str
    session_id: str
    source_timestamp: int | None


class FactDetails(BaseModel):
    linked_memory_ids: list[str]
    attributed_to: str | None


class ChunkResult(CommonHit):
    type: Literal["chunk"] = "chunk"
    chunk: ChunkDetails
    retrieval_sources: list[Literal["chunk", "fact"]] = Field(
        default_factory=list
    )
    supporting_memory_ids: list[str] = Field(default_factory=list)


class FactResult(CommonHit):
    type: Literal["fact"] = "fact"
    fact: FactDetails


class SearchResponse(BaseModel):
    data: list[
        Annotated[ChunkResult | FactResult, Field(discriminator="type")]
    ]
