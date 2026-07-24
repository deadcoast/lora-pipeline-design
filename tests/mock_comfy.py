"""A mock ComfyUI server implementing the real API contract.

Lets the integration test exercise the ACTUAL client + validation code offline —
no GPU, no models, no real ComfyUI. Implements the endpoints the pipeline uses:
  GET  /system_stats                          (reachability)
  GET  /object_info/CheckpointLoaderSimple    (doctor: checkpoint list)
  POST /prompt                                 (queue a workflow)
  WS   /ws                                     (execution progress -> done signal)
  GET  /history/{prompt_id}                    (outputs -> images)
  GET  /view                                   (image bytes)

Run standalone:  python -m tests.mock_comfy   (serves on 127.0.0.1:8765)
"""

import asyncio
import io
import uuid

from aiohttp import WSMsgType, web
from PIL import Image

CHECKPOINTS = ["sd_xl_base_1.0.safetensors", "sd_xl_refiner_1.0.safetensors"]
ws_clients: set[web.WebSocketResponse] = set()


async def system_stats(request):
    return web.json_response({"system": {"comfyui_version": "mock-1.0"}})


async def object_info(request):
    return web.json_response(
        {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [CHECKPOINTS]}}
            }
        }
    )


async def prompt(request):
    body = await request.json()
    assert "prompt" in body and "client_id" in body, "client must send prompt+client_id"
    pid = str(uuid.uuid4())
    prefix = "output"
    for node in body["prompt"].values():
        if node.get("class_type") == "SaveImage":
            prefix = node["inputs"].get("filename_prefix", "output")
    request.app["jobs"][pid] = prefix

    async def emit():
        await asyncio.sleep(0.03)
        for c in list(ws_clients):
            await c.send_json(
                {"type": "executing", "data": {"node": "3", "prompt_id": pid}}
            )
            await c.send_json({"type": "execution_success", "data": {"prompt_id": pid}})

    asyncio.create_task(emit())
    return web.json_response({"prompt_id": pid, "number": 1, "node_errors": {}})


async def history(request):
    pid = request.match_info["pid"]
    prefix = request.app["jobs"].get(pid, "output")
    return web.json_response(
        {
            pid: {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": f"{prefix}_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        }
    )


async def view(request):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 140, 200)).save(buf, "PNG")
    return web.Response(body=buf.getvalue(), content_type="image/png")


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
    finally:
        ws_clients.discard(ws)
    return ws


def make_app():
    app = web.Application()
    app["jobs"] = {}
    app.add_routes(
        [
            web.get("/system_stats", system_stats),
            web.get("/object_info/CheckpointLoaderSimple", object_info),
            web.post("/prompt", prompt),
            web.get("/history/{pid}", history),
            web.get("/view", view),
            web.get("/ws", ws_handler),
        ]
    )
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="127.0.0.1", port=8765)
