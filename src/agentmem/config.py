"""Validated configuration for storage, model services and retrieval channels."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatabaseConfig(ConfigModel):
    path: Path = Path("data/agentmem-v6.db")


class APIConfig(ConfigModel):
    require_token: bool = True
    api_key: SecretStr | None = None


class ServiceConfig(ConfigModel):
    base_url: str
    model: str
    api_key: SecretStr | None = None


class EmbeddingConfig(ServiceConfig):
    # Identity must describe the effective model and embedding input transform.
    identity: str = Field(min_length=1)
    request_timeout_seconds: float = Field(default=150, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=0, ge=0, strict=True)


class ChannelsConfig(ConfigModel):
    chunk: bool = True
    fact: bool = True


class RepairsConfig(ConfigModel):
    message_context: bool = True
    grounded_extraction: bool = True


class DiagnosticsConfig(ConfigModel):
    enabled: bool = False
    path: Path = Path("diagnostics/events.jsonl")
    include_content: bool = False


class RetrievalConfig(ConfigModel):
    mode: Literal["evidence", "legacy_mixed"] = "evidence"
    fact_assist: bool = True
    chunk_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    fact_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)


class FactExtractionConfig(ConfigModel):
    source_repair: bool = True
    response_format: Literal["json_object", "json_schema"] = "json_object"
    max_tokens: int = Field(default=2000, ge=1, strict=True)


class FactRetryConfig(ConfigModel):
    # Like SDK max_retries: 2 means one initial call plus two retries.
    max_retries: int = Field(default=2, ge=0, le=10, strict=True)
    backoff_seconds: float = Field(default=2, ge=0, allow_inf_nan=False)


class Settings(ConfigModel):
    api: APIConfig = Field(default_factory=APIConfig)
    fact_extraction: FactExtractionConfig = Field(default_factory=FactExtractionConfig)
    fact_retry: FactRetryConfig = Field(default_factory=FactRetryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: ServiceConfig | None = None
    embedding: EmbeddingConfig | None = None
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    repairs: RepairsConfig = Field(default_factory=RepairsConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    @model_validator(mode="after")
    def validate_retrieval(self):
        if self.retrieval.mode == "evidence" and not self.channels.chunk:
            raise ValueError("evidence retrieval requires channels.chunk=true")
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> "Settings":
        with Path(path).open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))
