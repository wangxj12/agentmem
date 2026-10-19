"""FastAPI boundary, kept separate from channel algorithms and HTTP clients."""

from contextlib import asynccontextmanager
from secrets import compare_digest

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .auth import resolve_api_key
from .config import Settings
from .evidence_context import with_evidence_context
from .requests import RequestConflictError
from .schemas import AddRequest, AddResponse, ChunkMetadataRequest, SearchRequest, SearchResponse


def create_app(service=None, settings=None):
    settings = settings if settings is not None else Settings()
    api_key = resolve_api_key(settings.api)

    def require_token(request: Request):
        if api_key is None:
            return
        scheme, separator, supplied = request.headers.get("Authorization", "").partition(" ")
        if not (separator and scheme.lower() == "token"
                and compare_digest(supplied.encode("utf-8"), api_key.encode("ascii"))):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API token",
                headers={"WWW-Authenticate": "Token"},
            )

    router = APIRouter(dependencies=[Depends(require_token)])

    @asynccontextmanager
    async def lifespan(app):
        if service is None:
            from .runtime import build_runtime

            runtime = build_runtime(settings)
            app.state.service = runtime.service
            try:
                yield
            finally:
                runtime.close()
        else:
            yield

    app = FastAPI(
        title="agentmem",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    if service is not None:
        app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: error.get(key) for key in ("loc", "type", "msg")}
                    for error in exc.errors()
                ]
            },
        )

    @router.post("/v1/memories/add", response_model=AddResponse)
    @router.post("/add", response_model=AddResponse, include_in_schema=False)
    def add(payload: AddRequest, request: Request):
        try:
            return request.app.state.service.add(payload)
        except RequestConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post("/search", response_model=SearchResponse)
    def search(payload: SearchRequest, request: Request):
        return request.app.state.service.search(payload)

    @router.post("/v1/memories/search", response_model=SearchResponse)
    @router.post("/search/aml", response_model=SearchResponse, include_in_schema=False)
    def search_aml(payload: SearchRequest, request: Request):
        return with_evidence_context(request.app.state.service.search(payload))

    @router.get('/status')
    def status(request: Request):
        return request.app.state.service.info()

    @router.get('/requests/{request_id}')
    def request_status(request_id: str, request: Request):
        value = request.app.state.service.states.get(request_id)
        if value is None:
            raise HTTPException(status_code=404, detail='Request status not found')
        return value

    @router.post('/chunks/metadata')
    def chunk_metadata(payload: ChunkMetadataRequest, request: Request):
        if not payload.ids:
            return {'data': []}
        store = request.app.state.service.chunk.store
        ids = [value.removeprefix('chunk:') for value in payload.ids]
        placeholders = ','.join('?' for _ in ids)
        with store.connection() as conn:
            rows = conn.execute(
                f"""SELECT id,request_id,message_index,start_char,end_char,role,speaker,source_timestamp,session_id
                    FROM raw_chunks WHERE user_id=? AND id IN ({placeholders})""",
                (payload.user_id, *ids),
            ).fetchall()
        return {'data': [dict(row) for row in rows]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(router)
    return app
