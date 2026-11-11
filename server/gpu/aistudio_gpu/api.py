"""The job API (FastAPI). Routes:

    GET  /v1/health                  no auth; liveness
    GET  /v1/info                    version, tasks (GPU, price, installed), never starts a GPU
    HEAD /v1/files/{name}            200 stored | 404 + X-Upload-Offset
    PUT  /v1/files/{name}            resumable, verified upload (201 / 202 / 409)
    DELETE /v1/files/{name}          removes an upload (a LoRA removed from the app's library) → 204 / 404
    POST /v1/jobs                    {"task", "params", "inputs": {"image": "<sha>.png"}} → 201 job summary
    GET  /v1/jobs/{id}               job summary
    WS   /v1/jobs/{id}/events?after= binary WireEvent records (see wire.py); closes after the terminal record
    GET  /v1/jobs/{id}/result        the result file (Range, ETag)
    POST /v1/jobs/{id}/cancel        cancels; billing stops with the GPU work
    POST /v1/fetch                   {"url", "sha256"?, "bytes"?, "ext"?}: the server downloads a file (a LoRA from
                                     Civitai / Hugging Face) into its uploads → fetch summary (see fetch.py)
    GET  /v1/fetch/{id}              its progress: state, bytes, total, name, error
    POST /v1/fetch/check             {"url", "sha256"?}: could the server download it? {stored, ok, error} (previews)
    DELETE /v1/fetch/{id}            cancels it
    POST /v1/models/import           {"family", "id", "name", "files": [{"url", "sha256", "role"}] | "repo"}: installs a
                                     fine-tune from a link (variants.py) → its import state
    GET  /v1/models                  {"families": {family: [{id, name, bytes}]}, "imports": {"family/id": state}}
    DELETE /v1/models/{family}/{id}  removes a fine-tune → 204 / 404
    /v1/llm/…                        LLMs from links and OpenAI-compatible chat (llm/routes.py), when the server has
                                     them (``"llm": true`` in /v1/info)

Auth: on Modal, proxy auth at Modal's edge (requests never reach us without a token). Elsewhere, an optional bearer
token checked here (constant time), normally behind TLS (Caddy, tailscale serve).
"""

from __future__ import annotations

import asyncio
import hmac
import re

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .fetch import Fetcher
from .variants import Imports
from .files import FileStore, HashMismatch, OffsetMismatch, is_name
from .jobs import TERMINAL, JobManager, ModelsMissing
from .wire import Kind

UPLOAD_OFFSET = "X-Upload-Offset"
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


def create_app(manager: JobManager, token: str | None = None, fetcher: Fetcher | None = None,
               variants=None, llm=None) -> FastAPI:
    app = FastAPI(title="AIStudio GPU server", docs_url=None, redoc_url=None, openapi_url=None)
    files: FileStore = manager.files
    fetcher = fetcher or Fetcher(files)
    if variants is None and getattr(manager.runner, "models_root", None):
        variants = Imports(manager.runner.models_root, extra_hosts=tuple(fetcher.extra_hosts))

    def authorized(header: str | None) -> bool:
        if token is None:
            return True
        if not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:].strip().encode(), token.encode())

    @app.middleware("http")
    async def auth_and_headers(request: Request, call_next):
        if request.url.path != "/v1/health" and not authorized(request.headers.get("authorization")):
            response = JSONResponse({"error": "missing or wrong token"}, status_code=401)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
        return response

    def job_or_404(job_id: str):
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, e: HTTPException):
        return JSONResponse({"error": e.detail}, status_code=e.status_code, headers=e.headers)

    @app.get("/v1/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/info")
    async def info():
        return {**manager.info(), "llm": llm is not None}

    if llm is not None:
        from .llm.routes import llm_router

        app.include_router(llm_router(llm))

    @app.head("/v1/files/{name}")
    async def file_head(name: str):
        if not is_name(name):
            raise HTTPException(400, "bad file name")
        if files.exists(name):
            return Response(status_code=200)
        return Response(status_code=404, headers={UPLOAD_OFFSET: str(files.partial_size(name))})

    @app.delete("/v1/files/{name}")
    async def file_delete(name: str):
        if not is_name(name):
            raise HTTPException(400, "bad file name")
        if not await files.delete(name):
            raise HTTPException(404, "no such file")
        return Response(status_code=204)

    @app.put("/v1/files/{name}")
    async def file_put(name: str, request: Request):
        if not is_name(name):
            raise HTTPException(400, "bad file name")
        offset, total = 0, None
        if (length := request.headers.get("content-length")) is not None:
            total = int(length)
        if (cr := request.headers.get("content-range")) and (m := CONTENT_RANGE.match(cr)):
            offset, total = int(m.group(1)), int(m.group(3))
        try:
            complete = await files.receive(name, offset, total, request.stream())
        except OffsetMismatch as e:
            return Response(status_code=409, headers={UPLOAD_OFFSET: str(e.expected)})
        except HashMismatch as e:
            raise HTTPException(400, str(e)) from None
        size = total if complete and total is not None else files.partial_size(name)
        return Response(status_code=201 if complete else 202, headers={UPLOAD_OFFSET: str(size)})

    @app.post("/v1/fetch")
    async def fetch_start(request: Request):
        body = await request.json()
        try:
            f = await fetcher.start(str(body["url"]), body.get("sha256"), body.get("bytes"), body.get("ext") or "lora")
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, str(e)) from None
        return JSONResponse(f.summary(), status_code=200 if f.state == "done" else 202)

    @app.post("/v1/fetch/check")
    async def fetch_check(request: Request):
        body = await request.json()
        try:
            return await fetcher.probe(str(body["url"]), body.get("sha256"), body.get("ext") or "lora")
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/v1/fetch/{fetch_id}")
    async def fetch_status(fetch_id: str):
        f = fetcher.get(fetch_id)
        if f is None:
            raise HTTPException(404, "no such download")
        return f.summary()

    @app.delete("/v1/fetch/{fetch_id}")
    async def fetch_cancel(fetch_id: str):
        f = fetcher.cancel(fetch_id)
        if f is None:
            raise HTTPException(404, "no such download")
        return f.summary()

    @app.post("/v1/models/import")
    async def model_import(request: Request):
        if variants is None:
            raise HTTPException(501, "this server can't install models")
        body = await request.json()
        try:
            return await asyncio.to_thread(variants.start, str(body["family"]), str(body["id"]),
                                           str(body.get("name") or body["id"]), body.get("files"), body.get("repo"))
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/v1/models")
    async def model_list():
        if variants is None:
            return {"families": {}, "imports": {}}
        return await asyncio.to_thread(variants.listing)

    @app.delete("/v1/models/{family}/{vid}")
    async def model_delete(family: str, vid: str):
        if variants is None:
            raise HTTPException(501, "this server can't install models")
        try:
            removed = await asyncio.to_thread(variants.remove, family, vid)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        if not removed:
            raise HTTPException(404, "no such model")
        return Response(status_code=204)

    @app.post("/v1/jobs", status_code=201)
    async def submit(request: Request):
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "expected a JSON body") from None
        if not isinstance(body, dict) or not isinstance(body.get("task"), str):
            raise HTTPException(400, "expected {\"task\": …, \"params\": {…}, \"inputs\": {…}}")
        try:
            job = await manager.submit(body["task"], body.get("params") or {}, body.get("inputs") or {})
        except LookupError as e:
            raise HTTPException(400, str(e).strip("'\"")) from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except FileNotFoundError as e:
            raise HTTPException(404, f"input file {e.args[0]} isn't on the server; upload it first") from None
        except ModelsMissing as e:
            raise HTTPException(503, str(e)) from None
        return job.summary()

    @app.get("/v1/jobs/{job_id}")
    async def job_summary(job_id: str):
        return job_or_404(job_id).summary()

    @app.post("/v1/jobs/{job_id}/cancel", status_code=202)
    async def cancel(job_id: str):
        job = job_or_404(job_id)
        await manager.cancel(job)
        return job.summary()

    @app.get("/v1/jobs/{job_id}/result")
    async def result(job_id: str, request: Request):
        job = job_or_404(job_id)
        if job.result is None:
            raise HTTPException(409 if job.state not in TERMINAL else 404, f"job is {job.state}")
        r = job.result
        headers = {"ETag": f'"{r.etag}"', "Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
        start, end = 0, r.size - 1
        status = 200
        range_header = request.headers.get("range")
        if_range = request.headers.get("if-range")
        if range_header and (if_range is None or if_range.strip('"') == r.etag):
            m = RANGE.match(range_header.strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                raise HTTPException(416, "bad range", headers={"Content-Range": f"bytes */{r.size}"})
            if m.group(1) == "":  # suffix: last N bytes
                start = max(0, r.size - int(m.group(2)))
            else:
                start = int(m.group(1))
                end = min(int(m.group(2)), r.size - 1) if m.group(2) else r.size - 1
            if start > end or start >= r.size:
                raise HTTPException(416, "range not satisfiable", headers={"Content-Range": f"bytes */{r.size}"})
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{r.size}"
        headers["Content-Length"] = str(end - start + 1)

        def chunks():
            with open(r.path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    block = f.read(min(1 << 20, left))
                    if not block:
                        break
                    left -= len(block)
                    yield block

        return StreamingResponse(chunks(), status_code=status, media_type=r.content_type, headers=headers)

    @app.websocket("/v1/jobs/{job_id}/events")
    async def events(ws: WebSocket, job_id: str, after: int = 0):
        # Accept first, then close with a code the client can tell apart from a network error.
        await ws.accept()
        if not authorized(ws.headers.get("authorization")):
            await ws.close(code=4401)
            return
        job = manager.jobs.get(job_id)
        if job is None:
            await ws.close(code=4404)
            return
        backlog, q = manager.subscribe(job, after)
        try:
            if backlog:
                await ws.send_bytes(b"".join(backlog))
            done = False
            while not done:
                item = await q.get()
                batch = []
                # Coalesce whatever else is already waiting into the same frame; of several previews keep the last.
                while True:
                    if item is None:
                        done = True
                        break
                    batch.append(item)
                    try:
                        item = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                last_preview = max((i for i, (k, _) in enumerate(batch) if k == Kind.PREVIEW), default=-1)
                frame = b"".join(rec for i, (k, rec) in enumerate(batch) if k != Kind.PREVIEW or i == last_preview)
                if frame:
                    await ws.send_bytes(frame)
            await ws.close()
        except WebSocketDisconnect:
            pass
        finally:
            manager.unsubscribe(job, q)

    return app
