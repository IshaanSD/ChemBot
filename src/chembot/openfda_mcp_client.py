import os
import sys
import json
import asyncio
import threading
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class OpenFDAMCPClient:
    """Adapter client that mimics the old OpenFDAClient interface but routes queries

    to the local OpenFDA MCP server over Stdin/Stdout (stdio) transport.
    """
    def __init__(self, command: Optional[str] = None, args: Optional[List[str]] = None):
        # Configure connection params from arguments or environment variables
        self.command = command or os.environ.get("OPENFDA_MCP_COMMAND", "python")
        
        if args is None:
            args_env = os.environ.get("OPENFDA_MCP_ARGS")
            if args_env:
                self.args = args_env.split()
            else:
                # Default to launching the local MCP server script relative to the project root
                self.args = ["mcp_servers/openfda_server.py"]
        else:
            self.args = args

        self.server_params = StdioServerParameters(command=self.command, args=self.args)
        
        # We run the async MCP client inside a dedicated background thread with its own event loop
        # so that it can expose a clean, synchronous interface to existing LangGraph nodes.
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        self.session = None
        self.stdio_ctx = None
        self.session_ctx = None
        self._connect()

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _connect(self) -> None:
        # Run connection on the background loop and wait for it to complete
        future = asyncio.run_coroutine_threadsafe(self._async_connect(), self.loop)
        future.result()

    async def _async_connect(self) -> None:
        self.stdio_ctx = stdio_client(self.server_params)
        read_stream, write_stream = await self.stdio_ctx.__aenter__()
        self.session_ctx = ClientSession(read_stream, write_stream)
        self.session = await self.session_ctx.__aenter__()
        await self.session.initialize()

    def close(self) -> None:
        if self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._async_close(), self.loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)

    async def _async_close(self) -> None:
        if self.session_ctx:
            await self.session_ctx.__aexit__(None, None, None)
        if self.stdio_ctx:
            await self.stdio_ctx.__aexit__(None, None, None)

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(name, arguments), self.loop
        )
        return future.result()

    async def _async_call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if os.environ.get("CHEMBOT_DEBUG") == "1":
            print(f"\n[MCP CLIENT DEBUG] Calling tool '{name}' with arguments: {arguments}", file=sys.stderr)
            
        result = await self.session.call_tool(name, arguments=arguments)
        
        if os.environ.get("CHEMBOT_DEBUG") == "1":
            content_preview = str(result.content)[:400] + "..." if len(str(result.content)) > 400 else str(result.content)
            print(f"[MCP CLIENT DEBUG] Raw tool result content preview: {content_preview}", file=sys.stderr)
            
        if not result.content:
            return {"results": [], "error": "No content returned from tool"}
        
        content_block = result.content[0]
        # FastMCP serializes dictionary return values into JSON strings inside TextContent blocks
        if getattr(content_block, "type", None) == "text" and hasattr(content_block, "text"):
            try:
                parsed = json.loads(content_block.text)
                if os.environ.get("CHEMBOT_DEBUG") == "1":
                    results_len = len(parsed.get("results", []))
                    info = parsed.get("info") or parsed.get("error") or "Success"
                    print(f"[MCP CLIENT DEBUG] Parsed tool response: found {results_len} results. Status: {info}", file=sys.stderr)
                return parsed
            except json.JSONDecodeError:
                return {"results": [], "error": f"Failed to parse JSON response: {content_block.text}"}
        
        return {"results": [], "error": f"Unexpected content type from tool: {content_block}"}

    def search_labels(self, search: str, limit: int = 3, skip: int = 0) -> List[Dict[str, Any]]:
        res = self._call_tool("search_drug_labels", {"search": search, "limit": limit, "skip": skip})
        if "error" in res:
            # Log the error, or simply return empty list to maintain current error-tolerance behaviour
            # of ChemBot which defaults to empty list on network or empty matches.
            return []
        return res.get("results", [])

    def search_events(self, search: str, limit: int = 3, skip: int = 0) -> List[Dict[str, Any]]:
        res = self._call_tool("search_adverse_events", {"search": search, "limit": limit, "skip": skip})
        if "error" in res:
            return []
        return res.get("results", [])

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
