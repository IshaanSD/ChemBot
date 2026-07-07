import os
import sys
import json
import asyncio
import threading
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class PubMedMCPClient:
    """Adapter client that routes PubMed queries to the local PubMed MCP server
    over Stdin/Stdout (stdio) transport.

    Follows the same thread-safe async-in-background pattern as OpenFDAMCPClient.
    """

    def __init__(self, command: Optional[str] = None, args: Optional[List[str]] = None):
        self.command = command or os.environ.get("PUBMED_MCP_COMMAND", "python")

        if args is None:
            args_env = os.environ.get("PUBMED_MCP_ARGS")
            if args_env:
                self.args = args_env.split()
            else:
                self.args = ["mcp_servers/pubmed_server.py"]
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

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _connect(self) -> None:
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
        debug = os.environ.get("CHEMBOT_DEBUG") == "1"
        if debug:
            print(f"\n[PUBMED MCP DEBUG] Calling tool '{name}' with arguments: {arguments}", file=sys.stderr)

        result = await self.session.call_tool(name, arguments=arguments)

        if debug:
            content_preview = str(result.content)[:400] + "..." if len(str(result.content)) > 400 else str(result.content)
            print(f"[PUBMED MCP DEBUG] Raw tool result content preview: {content_preview}", file=sys.stderr)

        if not result.content:
            return {"results": [], "error": "No content returned from tool"}

        content_block = result.content[0]
        if getattr(content_block, "type", None) == "text" and hasattr(content_block, "text"):
            try:
                parsed = json.loads(content_block.text)
                if debug:
                    results_len = len(parsed.get("results", []))
                    print(f"[PUBMED MCP DEBUG] Parsed tool response: found {results_len} results.", file=sys.stderr)
                return parsed
            except json.JSONDecodeError:
                return {"results": [], "error": f"Failed to parse JSON response: {content_block.text}"}

        return {"results": [], "error": f"Unexpected content type from tool: {content_block}"}

    # ─── Public API ──────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """General PubMed search."""
        res = self._call_tool("search_pubmed", {"query": query, "max_results": max_results})
        return res.get("results", [])

    def search_clinical_trials(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for clinical trial publications."""
        res = self._call_tool("search_clinical_trials", {"query": query, "max_results": max_results})
        return res.get("results", [])

    def search_systematic_reviews(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for systematic review publications."""
        res = self._call_tool("search_systematic_reviews", {"query": query, "max_results": max_results})
        return res.get("results", [])

    def search_meta_analyses(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for meta-analysis publications."""
        res = self._call_tool("search_meta_analyses", {"query": query, "max_results": max_results})
        return res.get("results", [])

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch PubMed article details by PMID list."""
        res = self._call_tool("fetch_abstracts", {"pmids": pmids})
        return res.get("results", [])

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
