"""Minimal HTTP adapters; model/input transforms are part of database identity."""

import os
from types import SimpleNamespace

import numpy as np
from openai import BadRequestError, OpenAI, UnprocessableEntityError

from ._mem0.llm_parameters import LLMParameters
from .config import FactExtractionConfig
from .diagnostics import emit, observed


class HTTPEmbedding:
    def __init__(self, config, channel, *, client=None):
        self.config, self.channel = config, channel
        transform = {
            "chunk": "plain-l2-f32",
            "fact": "newline-space-faiss-cosine",
        }[channel]
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if channel == "chunk"
            and "bge" in config.model.lower()
            and "-en-" in config.model.lower()
            else ""
        )
        transform += f"|query_prefix={self.query_prefix}"
        self.identity = f"{config.identity}|model={config.model}|input={transform}"
        self.key = (
            config.api_key.get_secret_value()
            if config.api_key
            else os.environ.get("AGENTMEM_EMBEDDING_API_KEY", "not-required")
        )
        self.client = client or OpenAI(
            api_key=self.key,
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )

    @observed("embedding_http")
    def encode_documents(self, texts):
        emit(
            "embedding_input",
            channel=self.channel,
            count=len(texts),
            content={"texts": texts},
        )
        inputs = (
            [text.replace("\n", " ") for text in texts]
            if self.channel == "fact"
            else texts
        )
        vectors = []
        batch_size = 64 if self.channel == "chunk" else 100
        for i in range(0, len(inputs), batch_size):
            response = self.client.embeddings.create(
                model=self.config.model,
                input=inputs[i : i + batch_size],
                encoding_format="float",
            )
            vectors.extend(
                item.embedding
                for item in sorted(response.data, key=lambda item: item.index)
            )
        if len(vectors) != len(texts):
            raise ValueError("Embedding response count differs from input count")
        emit(
            "embedding_output",
            channel=self.channel,
            count=len(vectors),
            content={"vectors": vectors},
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.channel == "chunk" and len(matrix):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix

    def encode_queries(self, texts):
        return self.encode_documents([self.query_prefix + text for text in texts])

    def close(self):
        self.client.close()


class FactLLM(LLMParameters):
    def __init__(self, config, *, client=None, extraction_config=None):
        self.extraction_config = extraction_config or FactExtractionConfig()
        self.config = SimpleNamespace(
            model=config.model,
            temperature=0.0,
            max_tokens=self.extraction_config.max_tokens,
            top_p=0.1,
            is_reasoning_model=None,
        )
        key = (
            config.api_key.get_secret_value()
            if config.api_key
            else os.environ.get("AGENTMEM_LLM_API_KEY", "not-required")
        )
        self.client = client or OpenAI(api_key=key, base_url=config.base_url)

    @observed("llm_http", channel="fact")
    def generate_response(
        self, messages, response_format=None, *, purpose="extraction", **kwargs
    ):
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update(model=self.config.model, messages=messages)
        if response_format:
            params["response_format"] = response_format
        requested = (response_format or {}).get("type")
        fallback = False
        try:
            response = self.client.chat.completions.create(**params)
        except (BadRequestError, UnprocessableEntityError) as error:
            # Only a server rejection specifically naming structured output is
            # eligible. Auth/rate-limit and unrelated invalid parameters propagate.
            detail = str(getattr(error, "body", "")).lower()
            if requested != "json_schema" or not any(
                term in detail
                for term in ("json_schema", "response_format", "structured output")
            ):
                raise
            emit(
                "fact_schema_fallback",
                channel="fact",
                http_status=error.status_code,
                reason="server_rejected_schema",
                effective_format="json_object",
            )
            params["response_format"] = {"type": "json_object"}
            fallback = True
            response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        tokens = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            tokens[key] = value if type(value) is int else None
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None)
        tokens["reasoning_tokens"] = reasoning if type(reasoning) is int else None
        finish = getattr(choice, "finish_reason", None)
        emit(
            "fact_llm_metadata",
            channel="fact",
            phase=purpose if purpose in ("extraction", "source_repair") else "unknown",
            finish_reason=finish
            if finish
            in (None, "stop", "length", "tool_calls", "content_filter", "function_call")
            else "unknown",
            token_usage=tokens,
            requested_format=requested,
            effective_format=params.get("response_format", {}).get("type"),
            schema_fallback=fallback,
            local_protocol_check=True,
        )
        return choice.message.content

    def close(self):
        self.client.close()
