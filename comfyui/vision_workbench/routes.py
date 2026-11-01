"""Read-only same-origin proxies; camera frames remain in the Workbench viewer."""
from __future__ import annotations

import asyncio
from aiohttp import web
from . import client


def register_routes(server):
    if getattr(server, "_vision_workbench_routes", False):
        return
    server._vision_workbench_routes = True

    @server.routes.get("/vision-workbench/templates")
    async def templates(request):
        try:
            result = await asyncio.to_thread(client.request_json, "GET", "/api/comfy/templates")
            return web.json_response(result)
        except (RuntimeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=503)

    @server.routes.get("/vision-workbench/resources")
    async def resources(request):
        result = await asyncio.to_thread(client.resources, refresh=True)
        fields = {
            "models": ("id", "name", "task", "available", "inference_supported"),
            "checkpoints": ("id", "name", "model_id", "valid"),
            "media": ("id", "name", "kind"),
        }
        clean = {key: [{field: item.get(field) for field in selected} for item in result[key]]
                 for key, selected in fields.items()}
        clean["available"] = result["available"]
        return web.json_response(clean)
