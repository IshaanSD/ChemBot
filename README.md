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

## Run ChemBot (Default MCP Mode)

By default, ChemBot runs using the Model Context Protocol (MCP) data access layer:

```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db
```

To switch models:
```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db --openai_model gpt-4-mini
```

To run using the legacy direct HTTP fallback mode instead of MCP:
```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db --use_direct_openfda
```

---

## Running ChemBot with OpenFDA MCP

Follow these steps to run ChemBot using the Model Context Protocol (MCP) server/client architecture:

### 1. Install requirements
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
Export your OpenAI credentials:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Optionally set openFDA API Key
For higher query limits, you can optionally export your openFDA credentials:
```bash
export OPENFDA_API_KEY="your-openfda-key"
```

### 4. Start the OpenFDA MCP Server
*   **Stdio Transport (Default):** You do **not** need to manually start the server. The ChemBot client automatically spawns, connects to, and handles the local MCP server subprocess on launch.
*   **Dev Mode / Standalone:** If you want to run the server in standalone debug mode to view tool registration and schemas, run:
    ```bash
    fastmcp dev mcp_servers/openfda_server.py
    ```

### 5. Run ChemBot
Start the CLI agent (which runs in MCP mode by default):
```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db
```

### 6. Example Label Question
Once ChemBot starts, ask about a medication label:
```
User: What are the warnings for Aspirin?
```

### 7. Example Adverse Event Question
Ask about adverse safety events:
```
User: What side effects are reported for Boniva?
```

### 8. How to verify the MCP server is actually being used
Enable debug logging to see the live JSON-RPC traffic between the client and the MCP server. Set the `CHEMBOT_DEBUG=1` environment variable before running:
```bash
export CHEMBOT_DEBUG=1
python -m chembot.graph_cli --chroma_dir ./chroma_db
```
You will see output prefixed with `[MCP CLIENT DEBUG]` in standard error as the tools are called:
```
[MCP CLIENT DEBUG] Calling tool 'search_drug_labels' with arguments: {'search': '...', 'limit': 3}
[MCP CLIENT DEBUG] Raw tool result content preview: [TextContent(type='text', text='...')]
[MCP CLIENT DEBUG] Parsed tool response: found 1 results. Status: Success
```

---

## Configuration & Debugging

### Custom MCP Server Invocation
By default, the client launches the local MCP server using `python mcp_servers/openfda_server.py`. You can customize the command and arguments via environment variables:
*   `OPENFDA_MCP_COMMAND`: The executable (defaults to `python`).
*   `OPENFDA_MCP_ARGS`: Command arguments (defaults to `mcp_servers/openfda_server.py`).