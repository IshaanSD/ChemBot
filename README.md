# ChemBot

ChemBot is a CLI medication-label Q&A agent built with:
- **LangGraph** (graph-based agent control loop)
- **Model Context Protocol (MCP)** (server/client architecture wrapping openFDA)
- **ChromaDB** (persistent local vector store)
- **Sentence Transformers** (local embeddings)
- **OpenAI** (LLM responses)

ChemBot answers using ONLY retrieved FDA drug label excerpts.

---

## Architecture

ChemBot decouples the openFDA API data retrieval into a dedicated MCP server. Communication occurs via JSON-RPC over Standard Input/Output (stdio) transport:

```
[LangGraph Node] 
       │ (sync calls)
       ▼
[OpenFDAMCPClient] (Client Adapter)
       │ (async event loop thread)
       ▼
[Stdio JSON-RPC Transport]
       │
       ▼
[FastMCP Server] (mcp_servers/openfda_server.py)
       │ (requests)
       ▼
[openFDA HTTP API]
```

---

## Install ChemBot

```bash
# Change directory to ChemBot
pip install -r requirements.txt

# Export OpenAI credential
export OPENAI_API_KEY="your-openai-api-key"
```

---

## Run ChemBot

```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db
```

To run with a different OpenAI model:

```bash
python -m chembot.graph_cli \
  --chroma_dir ./chroma_db \
  --openai_model gpt-4-mini
```

---

## Configuration & Debugging

### (Optional) Use openFDA API key
For higher request limits (up to 120,000 requests per day), obtain a free openFDA API key and pass it via the CLI:
```bash
python -m chembot.graph_cli \
  --chroma_dir ./chroma_db \
  --openfda_api_key YOUR_OPENFDA_KEY
```
*(Alternatively, you can export `OPENFDA_API_KEY="YOUR_KEY"` into your environment)*

### View MCP Client Communication Logs
To see JSON-RPC inputs, outputs, and status information from the MCP connection printed to standard error, export `CHEMBOT_DEBUG=1`:
```bash
export CHEMBOT_DEBUG=1
python -m chembot.graph_cli --chroma_dir ./chroma_db
```

### Custom MCP Server Invocation
By default, the client launches the local MCP server using `python mcp_servers/openfda_server.py`. You can customize the command and arguments via environment variables:
*   `OPENFDA_MCP_COMMAND`: The executable (defaults to `python`).
*   `OPENFDA_MCP_ARGS`: Command arguments (defaults to `mcp_servers/openfda_server.py`).