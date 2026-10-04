"""
MCP-style JSON-RPC server exposing the send_email tool.

Endpoints:
  GET  /health   -> {"status":"ok"}
  POST /mcp      -> JSON-RPC 2.0 with methods: initialize, tools/list, tools/call
"""
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .email_tool import send_email

app = FastAPI(title="Movie Assistant MCP Server")

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "send_email",
        "title": "Send an Email",
        "description": "Send an email with a given subject and body to a recipient.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body (plain text)"},
            },
            "required": ["to", "subject", "body"],
        },
    }
]


def _ok(req_id: Any, result: Dict) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return _err(None, -32700, "Parse error")

    method = payload.get("method")
    req_id = payload.get("id")
    params = payload.get("params", {}) or {}

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "movie-assistant-mcp", "version": "1.0.0"},
        })

    if method == "tools/list":
        return _ok(req_id, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name != "send_email":
            return _ok(req_id, {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            })

        result = send_email(
            recipient_email=args.get("to", ""),
            subject=args.get("subject", ""),
            body=args.get("body", ""),
        )
        is_error = result.startswith("Error")
        return _ok(req_id, {
            "content": [{"type": "text", "text": result}],
            "isError": is_error,
        })

    return _err(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)