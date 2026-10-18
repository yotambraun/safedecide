"""Minimal ``/v1/systemone`` server, drop-in for clients built for Jev."""

from __future__ import annotations

import os
import secrets
from typing import Any


def create_app(model_path: str):
    from fastapi import FastAPI, Header, HTTPException

    from . import __version__
    from .predict import SafeDecider

    decider = SafeDecider.from_pretrained(model_path)
    app = FastAPI(title="safedecide", version=__version__)
    api_key = os.environ.get("SAFEDECIDE_API_KEY")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": decider.config.get("run_name", model_path)}

    @app.post("/v1/systemone")
    def systemone(
        body: dict[str, Any], alpha: float | None = None, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        """``?alpha=0.1`` picks one of the model's certified error levels."""
        if api_key and not secrets.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(401, "invalid api key")
        try:
            return decider.decide(body, alpha=alpha)
        except (KeyError, ValueError) as e:
            raise HTTPException(422, str(e)) from e

    return app


def serve(model_path: str, host: str = "127.0.0.1", port: int = 8177) -> None:
    import uvicorn

    uvicorn.run(create_app(model_path), host=host, port=port)
