"""Construction only; no service call runs on import or startup."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .chunks import ChunkChannel
from .config import Settings
from .diagnostics import Diagnostics, emit
from .evidence_context import SEARCH_CONTENT_FORMAT
from .facts import FactChannel
from .http_clients import FactLLM, HTTPEmbedding
from .service import MemoryService
from .store import SCHEMA_VERSION, SQLiteStore


@dataclass
class Runtime:
    service: MemoryService
    clients: list

    def close(self):
        for client in self.clients:
            client.close()


def build_runtime(settings: Settings):
    if settings is None or settings.embedding is None:
        raise ValueError(
            "An embedding configuration is required for the mandatory chunk write"
        )
    secrets = [
        os.getenv("AGENTMEM_API_KEY"),
        os.getenv("AGENTMEM_LLM_API_KEY"),
        os.getenv("AGENTMEM_EMBEDDING_API_KEY"),
    ]
    if settings.api.api_key is not None:
        secrets.append(settings.api.api_key.get_secret_value())
    for config in (settings.llm, settings.embedding):
        if config and config.api_key:
            secrets.append(config.api_key.get_secret_value())
    diagnostics = Diagnostics(
        settings.diagnostics.path if settings.diagnostics.enabled else None,
        include_content=settings.diagnostics.include_content,
        secrets=secrets,
    )
    with diagnostics.scope("startup"):

        def endpoint(config):
            if config is None:
                return None
            url = urlsplit(config.base_url)
            return {
                "model": config.model,
                "endpoint": f"{url.scheme}://{url.hostname}:{url.port or ''}{url.path}",
            }

        root = Path(__file__).parent
        emit(
            "runtime_configuration",
            schema_version=SCHEMA_VERSION,
            package_version=__version__,
            retrieval=settings.retrieval.model_dump(),
            aml_search_content_format=SEARCH_CONTENT_FORMAT,
            llm=endpoint(settings.llm),
            embedding=endpoint(settings.embedding),
            embedding_identity=settings.embedding.identity,
            repairs=settings.repairs.model_dump(),
            fact_extraction=settings.fact_extraction.model_dump(),
            fact_retry=settings.fact_retry.model_dump(),
            channels=settings.channels.model_dump(),
            source_hashes={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*.py"))
            },
        )
    clients = []
    try:
        chunk_embed = HTTPEmbedding(settings.embedding, "chunk")
        clients.append(chunk_embed)
        fact_embed = HTTPEmbedding(settings.embedding, "fact")
        clients.append(fact_embed)
        store = SQLiteStore(
            settings.database.path,
            {
                "chunk": chunk_embed.identity,
                "fact": fact_embed.identity,
                "entity": fact_embed.identity,
            },
        )
        fact_llm = None
        if settings.llm:
            fact_llm = FactLLM(settings.llm, extraction_config=settings.fact_extraction)
            clients.append(fact_llm)
        service = MemoryService(
            ChunkChannel(store, chunk_embed),
            FactChannel(
                store,
                fact_embed,
                fact_llm,
                message_context=settings.repairs.message_context,
                grounded_extraction=settings.repairs.grounded_extraction,
                extraction_config=settings.fact_extraction,
            ),
            channels=settings.channels.model_dump(),
            diagnostics=diagnostics,
            retrieval=settings.retrieval,
            fact_retry=settings.fact_retry,
            configuration_id=hashlib.sha256(
                str((settings.channels.model_dump(), settings.retrieval.model_dump(),
                     settings.repairs.model_dump(), settings.fact_extraction.model_dump(),
                     settings.fact_retry.model_dump(), settings.llm.model if settings.llm else None,
                     settings.llm.base_url if settings.llm else None, store.embedding_identities,
                     SEARCH_CONTENT_FORMAT)).encode()
            ).hexdigest(),
        )
        return Runtime(service, clients)
    except Exception:
        for client in clients:
            client.close()
        raise
