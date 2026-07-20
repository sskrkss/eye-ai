import logging
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.config import get_settings
from database.database import init_db, close_db
from routes.routes_registry import get_app_router

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    is_dev = settings.APP_ENV == "dev"
    init_db(drop_all=is_dev, with_test_data=is_dev)
    yield
    close_db()


def create_application() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(get_app_router())

    return app


app = create_application()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(
        'api:app',
        host='0.0.0.0',
        port=8080,
        reload=True,
        log_level="info"
    )
