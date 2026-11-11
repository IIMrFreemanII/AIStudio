"""The LLM routes, mounted on the front desk (same auth as the job API):

    POST   /v1/llm/inspect              {"url", "hfToken"?, "civitaiToken"?, "context"?, "quant"?} → facts (GGUF
                                         quantizations in "ggufOptions"), parser profile, GPU plan, suggested id
    POST   /v1/llm/install              {"url", "id"?, "name"?, "gpu"?, "count"?, "context"?, "quant"?, "toolParser"?,
                                         "reasoningParser"?, "engineArgs"?, "hfToken"?, "civitaiToken"?} → state (202)
    GET    /v1/llm/models               {"models": [manifest + status], "installs": {id: state}}
    PATCH  /v1/llm/models/{id}          {"gpu"?, "count"?, "context"?, "name"?, "toolParser"?, …} → manifest
    DELETE /v1/llm/models/{id}          stops and removes it → 204 / 404
    POST   /v1/llm/models/{id}/warm     starts it (returns at once) → status
    POST   /v1/llm/models/{id}/keepalive keeps it warm a while longer → status (+ "ignored" past the lease)
    POST   /v1/llm/models/{id}/stop     shuts it down now → status
    GET    /v1/llm/models/{id}/plan     its GPU options again (to change GPUs or context) and the current choice
    GET    /v1/llm/models/{id}/status   {state: cold|starting|loading|ready|failed, phase, progress, uptime, gpu,
                                         count, usdPerSecond, costUSD, …}
    GET    /v1/llm/oai/v1/models        OpenAI's model list (installed models)
    POST   /v1/llm/oai/v1/chat/completions  OpenAI chat completions (streamed as SSE; see proxy.py). Images and
                                         videos can be uploads (PUT /v1/files first): {"type": "video_url",
                                         "video_url": {"url": "upload:<sha256>.mp4"}} (404 when it isn't there)

Tokens in a body (Hugging Face, Civitai) are used for that request only; they are never stored or logged.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .facts import LinkError
from .service import LLMService


def llm_router(service: LLMService) -> APIRouter:
    r = APIRouter(prefix="/v1/llm")

    async def body_of(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "expected a JSON body") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "expected a JSON object")
        return body

    async def guarded(call):
        try:
            return await call
        except LinkError as e:
            raise HTTPException(400, str(e)) from None
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from None
        except LookupError as e:
            raise HTTPException(404, str(e).strip("'\"")) from None
        except (KeyError, TypeError) as e:
            raise HTTPException(400, f"missing or wrong field {e}") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @r.post("/inspect")
    async def inspect(request: Request):
        return await guarded(service.inspect(await body_of(request)))

    @r.post("/install", status_code=202)
    async def install(request: Request):
        return await guarded(service.install(await body_of(request)))

    @r.get("/models")
    async def models():
        return await service.listing()

    @r.patch("/models/{mid}")
    async def update(mid: str, request: Request):
        return await guarded(service.update(mid, await body_of(request)))

    @r.delete("/models/{mid}")
    async def delete(mid: str):
        if not await guarded(service.remove(mid)):
            raise HTTPException(404, f"no model {mid}")
        return Response(status_code=204)

    @r.post("/models/{mid}/warm")
    async def warm(mid: str):
        return await guarded(service.warm(mid))

    @r.post("/models/{mid}/keepalive")
    async def keepalive(mid: str):
        return await guarded(service.keepalive(mid))

    @r.post("/models/{mid}/stop")
    async def stop(mid: str):
        return await guarded(service.stop(mid))

    @r.get("/models/{mid}/plan")
    async def plan(mid: str, context: int | None = None):
        return await guarded(service.plan(mid, context))

    @r.get("/models/{mid}/status")
    async def status(mid: str):
        return await guarded(service.status(mid))

    @r.get("/oai/v1/models")
    async def oai_models():
        listing = await service.listing()
        return {"object": "list", "data": [{"id": m["id"], "object": "model", "owned_by": "aistudio",
                                            "created": int(m.get("created") or 0)}
                                           for m in listing["models"] if m.get("installed")]}

    @r.post("/oai/v1/chat/completions")
    async def chat(request: Request):
        body = await body_of(request)
        mid = body.get("model")
        if not isinstance(mid, str):
            raise HTTPException(400, "expected \"model\": one of GET /v1/llm/oai/v1/models")

        async def start():
            return service.chat(mid, body)

        streaming, chunks = await guarded(start())
        if streaming:
            return StreamingResponse(chunks, media_type="text/event-stream",
                                     headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        data = b""
        try:
            async for c in chunks:
                data += c
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, str(e)) from None
        return Response(data, media_type="application/json")

    return r
