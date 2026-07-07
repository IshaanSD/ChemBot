import os
import sys
import json
import asyncio
import threading
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class RxNormMCPClient:
    """Adapter client for the local RxNorm MCP server over Stdin/Stdout (stdio) transport."""
    
    def __init__(self, command: Optional[str] = None, args: Optional[List[str]] = None):
        self.command = command or os.environ.get("RXNORM_MCP_COMMAND", "python")
        if args is None:
            args_env = os.environ.get("RXNORM_MCP_ARGS")
            if args_env:
                self.args = args_env.split()
            else:
                self.args = ["mcp_servers/rxnorm_server.py"]
        else:
            self.args = args

        self.server_params = StdioServerParameters(command=self.command, args=self.args)
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        self.session = None
        self.stdio_ctx = None
        self.session_ctx = None
        self._connect()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _connect_async(self):
        async def do_connect():
            self.stdio_ctx = stdio_client(self.server_params)
            read_stream, write_stream = await self.stdio_ctx.__aenter__()
            self.session_ctx = ClientSession(read_stream, write_stream)
            self.session = await self.session_ctx.__aenter__()
            await self.session.initialize()
        future = asyncio.run_coroutine_threadsafe(do_connect(), self.loop)
        future.result()

    def _connect(self):
        self._connect_async()
        
    def _call_tool_sync(self, name: str, arguments: Dict[str, Any]) -> Any:
        if not self.session:
            raise RuntimeError("RxNorm MCP client is not connected.")
            
        async def do_call():
            result = await self.session.call_tool(name, arguments)
            if result.isError:
                raise RuntimeError(f"MCP Tool {name} returned error: {result.content}")
            
            text_result = result.content[0].text
            return text_result
        
        future = asyncio.run_coroutine_threadsafe(do_call(), self.loop)
        raw_result = future.result()
        if isinstance(raw_result, str):
            try:
                return json.loads(raw_result)
            except Exception:
                return raw_result
        return raw_result

    def search_drug(self, name: str, max_entries: int = 5) -> List[Dict[str, Any]]:
        return self._call_tool_sync("search_drug", {"name": name, "max_entries": max_entries})

    def lookup_generic_name(self, rxcui: str) -> List[str]:
        return self._call_tool_sync("lookup_generic_name", {"rxcui": rxcui})

    def lookup_brand_name(self, rxcui: str) -> List[str]:
        return self._call_tool_sync("lookup_brand_name", {"rxcui": rxcui})

    def normalize_drug_name(self, name: str) -> Dict[str, Any]:
        return self._call_tool_sync("normalize_drug_name", {"name": name})

    def close(self):
        async def do_close():
            if self.session_ctx:
                await self.session_ctx.__aexit__(None, None, None)
            if self.stdio_ctx:
                await self.stdio_ctx.__aexit__(None, None, None)
        if self.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(do_close(), self.loop)
            try:
                future.result(timeout=2.0)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2.0)
