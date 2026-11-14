"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from policy_rag.api.routes import router
from policy_rag.config import get_settings
from policy_rag.service import PolicyContainer

_WEB_INDEX_PATH = Path(__file__).resolve().parent.parent / "web" / "index.html"


def create_app(container: PolicyContainer | None = None) -> FastAPI:
    if container is None:
        container = PolicyContainer(settings=get_settings())

    app = FastAPI(
        title=container.settings.app_name,
        description="E-Commerce Policy Intelligent QA System API",
        version="0.1.0",
    )

    app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routes: /health at root, and /api/v1 prefix for chat
    app.include_router(router, prefix="")
    app.include_router(router, prefix=container.settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    async def workbench_ui() -> FileResponse:
        return FileResponse(_WEB_INDEX_PATH, media_type="text/html; charset=utf-8")

    return app


app = create_app()
