import os
from typing import Dict, List, Any

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")


class MCPClient:
    """Minimal MCP JSON-RPC client over HTTP."""

    def __init__(self, url: str = DEFAULT_URL, timeout: int = 15):
        self.url = url
        self.timeout = timeout
        self._id = 0

    def _rpc(self, method: str, params: Dict | None = None) -> Dict:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message", "MCP error"))
        return data["result"]

    def initialize(self) -> Dict:
        return self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "movie-assistant-agent", "version": "1.0.0"},
        })

    def list_tools(self) -> List[Dict]:
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict) -> Dict:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def is_reachable(self) -> bool:
        try:
            requests.get(self.url.rsplit("/", 1)[0] + "/health", timeout=3)
            return True
        except Exception:
            return False


def call_email_via_mcp(recipient_email: str, subject: str, body: str) -> str:
    """
    Convenience wrapper that returns a plain text result (or error string).
    Used by the agent so its tool response stays a string.
    """
    client = MCPClient()
    try:
        result = client.call_tool("send_email", {"to": recipient_email, "subject": subject, "body": body})
        content = result.get("content", [])
        text = " ".join(c.get("text", "") for c in content) or "Email tool returned no output."
        if result.get("isError"):
            return f"Error: {text}"
        return text
    except Exception as e:
        return f"Error: MCP call failed: {e}"