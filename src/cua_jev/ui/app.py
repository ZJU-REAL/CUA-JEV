from __future__ import annotations

import argparse
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import load_local_env
from .manager import TASK_CATALOG, RunManager


def create_app(root: str | Path | None = None, data: str | Path | None = None):
    project_root = Path(root or Path.cwd()).resolve()
    manager = RunManager(project_root, data)
    static = Path(__file__).with_name("static")
    demos = project_root / "artifacts" / "demos"
    demos.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        yield

    app = FastAPI(title="CUA-JEV Console", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.manager = manager
    app.state.csrf_token = token
    app.mount("/static", StaticFiles(directory=static), name="static")
    app.mount("/demos", StaticFiles(directory=demos), name="demos")
    curated_media = project_root / "website" / "media"
    windows_catalog = project_root / "website" / "windows_demos.json"
    if curated_media.is_dir():
        app.mount("/media", StaticFiles(directory=curated_media), name="media")

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "")
        hostname = urlsplit("http://" + host).hostname
        if hostname not in {"127.0.0.1", "localhost", "::1", "testserver"}:
            return JSONResponse({"detail": "console accepts local browser access only"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != f"http://{host}":
            return JSONResponse({"detail": "cross-origin access rejected"}, status_code=403)
        if request.method in {"POST", "PUT", "DELETE"}:
            if request.headers.get("x-cua-jev-csrf") != token:
                return JSONResponse({"detail": "CSRF token required"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self'; media-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    def index():
        page = "home.html" if len(windows_case_data()["cases"]) == 4 else "index.html"
        return (static / page).read_text(encoding="utf-8")

    def windows_case_data():
        if not windows_catalog.is_file():
            return {"schema_version": 2, "cases": []}
        return json.loads(windows_catalog.read_text(encoding="utf-8"))

    @app.get("/data/windows_demos.json")
    def windows_demos():
        return windows_case_data()

    @app.get("/preview.html", response_class=HTMLResponse)
    def preview():
        return (static / "home.html").read_text(encoding="utf-8")

    @app.get("/early-work.html", response_class=HTMLResponse)
    def early_work():
        return (static / "index.html").read_text(encoding="utf-8")

    @app.get("/api/bootstrap")
    def bootstrap():
        demo_files = {
            task: {
                label: f"/demos/{task}-{suffix}.mp4"
                for label, suffix in (("hybrid", "hybrid"), ("gui_only", "gui-only"))
                if (demos / f"{task}-{suffix}.mp4").is_file()
            }
            for task in TASK_CATALOG
        }
        return {
            "csrf": token,
            "tasks": TASK_CATALOG,
            "active_id": manager.active_id,
            "jev_configured": bool(os.getenv("TYPESAFE_API_KEY")),
            "platform": os.name,
            "demos": demo_files,
        }

    @app.get("/api/runs")
    def list_runs():
        return manager.list()

    @app.get("/api/benchmarks")
    def benchmarks(task: str | None = None):
        try:
            return manager.benchmarks(task)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/runs")
    async def create_run(request: Request):
        try:
            return manager.create(await request.json())
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return manager.detail(run_id)
        except KeyError as exc:
            raise HTTPException(404, detail="run not found") from exc

    @app.get("/api/runs/{run_id}/steps")
    def get_run_steps(run_id: str):
        try:
            return manager.steps(run_id)
        except KeyError as exc:
            raise HTTPException(404, detail="run not found") from exc

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str):
        try:
            return manager.stop(run_id)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/baselines")
    async def import_baseline(request: Request):
        try:
            return manager.import_baseline(await request.json())
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/baselines/{run_id}/usage")
    async def attach_baseline_usage(run_id: str, request: Request):
        try:
            return manager.attach_baseline_usage(run_id, await request.json())
        except KeyError as exc:
            raise HTTPException(404, detail="run not found") from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.post("/api/baselines/{run_id}/steps")
    async def attach_baseline_steps(run_id: str, request: Request):
        try:
            return manager.attach_baseline_steps(run_id, await request.json())
        except KeyError as exc:
            raise HTTPException(404, detail="run not found") from exc
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    return app


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="CUA-JEV local experiment console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the console only binds to localhost")
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("install cua-jev[ui] to run the local console") from None
    uvicorn.run(create_app(data=args.data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
