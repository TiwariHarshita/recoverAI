from fastapi import FastAPI

from app.api.webhooks import router as webhooks_router
from app.config import APP_NAME, APP_VERSION


app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.include_router(webhooks_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
