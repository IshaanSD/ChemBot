# ChemBot

ChemBot is a CLI biomedical Q&A agent built with:
- **LangGraph** (graph-based agent control loop with multi-route fan-out)
- **Model Context Protocol (MCP)** (server/client architecture wrapping openFDA and PubMed)
- **ChromaDB** (persistent local vector store)
- **Sentence Transformers** (local embeddings)
- **OpenAI** (LLM responses)

ChemBot answers using ONLY retrieved evidence from three sources:
- **FDA Drug Labels** — official regulatory prescribing information
- **FAERS** — FDA Adverse Event Reporting System data
- **PubMed** — published biomedical literature via NCBI Entrez

---

## Architecture

```
User
  │
  ▼
LangGraph Router (multi-route)
  │
  ├── FDA Label Retrieval ──► openFDA MCP Server
  │
  ├── FAERS Event Retrieval ─► openFDA MCP Server
  │
  └── PubMed Retrieval ─────► PubMed MCP Server
  │
  ▼
Evidence Fusion (groups by source)
  │
  ▼
Answer Generation (OpenAI)
```

The router can select **one or more branches** per query:

| Question | Routes |
|----------|--------|
| "What are the contraindications for ibuprofen?" | FDA Label |
| "What adverse events have been reported for Ozempic?" | FAERS |
| "What recent studies exist on semaglutide and Alzheimer's?" | PubMed |
| "Is ibuprofen safe during pregnancy?" | FDA Label + PubMed |
| "Is semaglutide associated with pancreatitis?" | FAERS + PubMed |

### MCP Servers

Both MCP servers use **stdio transport** — ChemBot automatically spawns and manages them as subprocesses:

```
[LangGraph Node]
       │ (sync calls)
       ▼
[MCP Client Adapter] (thread-safe async event loop)
       │ (JSON-RPC over stdio)
       ▼
[FastMCP Server]
       │ (requests)
       ▼
[External API]
```

| Server | File | API | Tools |
|--------|------|-----|-------|
| OpenFDA | `mcp_servers/openfda_server.py` | openFDA | `search_drug_labels`, `search_adverse_events` |
| PubMed | `mcp_servers/pubmed_server.py` | NCBI Entrez | `search_pubmed`, `search_clinical_trials`, `search_systematic_reviews`, `search_meta_analyses`, `fetch_abstracts` |

---

## Install ChemBot

```bash
pip install -r requirements.txt

# Export OpenAI credential
export OPENAI_API_KEY="your-openai-api-key"

# NCBI requires an email for Entrez API usage (PubMed)
export NCBI_EMAIL="your-email@example.com"
```

### Optional API keys

```bash
# Higher openFDA rate limits
export OPENFDA_API_KEY="your-openfda-key"

# Higher NCBI rate limits (3 → 10 req/sec)
export NCBI_API_KEY="your-ncbi-key"
```

---

## Run ChemBot

By default, ChemBot runs with all three evidence sources via MCP:

```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db
```

### Disable PubMed

```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db --disable_pubmed
```

### Use legacy direct OpenFDA client (no MCP)

```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db --use_direct_openfda --disable_pubmed
```

### Switch LLM model

```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db --openai_model gpt-4-mini
```

---

## Example Questions

### FDA Label query
```
User: What are the contraindications for ibuprofen?
```

### FAERS adverse event query
```
User: What adverse events have been reported for Ozempic?
```

### PubMed literature query
```
User: What recent studies exist on semaglutide and Alzheimer's disease?
```

### Multi-source query (FDA Label + PubMed)
```
User: Is ibuprofen safe during pregnancy?
```

---

## Debug Logging

Enable debug logging to see MCP JSON-RPC traffic:

```bash
export CHEMBOT_DEBUG=1
python -m chembot.graph_cli --chroma_dir ./chroma_db
```

You will see output prefixed with `[MCP CLIENT DEBUG]` and `[PUBMED MCP DEBUG]` in stderr:
```
[MCP CLIENT DEBUG] Calling tool 'search_drug_labels' with arguments: {'search': '...', 'limit': 3}
[MCP CLIENT DEBUG] Parsed tool response: found 3 results. Status: Success
[PUBMED MCP DEBUG] Calling tool 'search_pubmed' with arguments: {'query': '...', 'max_results': 5}
[PUBMED MCP DEBUG] Parsed tool response: found 4 results.
```

---

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--chroma_dir` | (required) | Path to ChromaDB directory |
| `--collection` | `chembot_cache` | Chroma collection name |
| `--st_model` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `--k` | `6` | Top-k chunks for answer generation |
| `--label_limit` | `3` | Max FDA label records |
| `--event_limit` | `3` | Max FAERS event records |
| `--pubmed_limit` | `5` | Max PubMed abstracts |
| `--disable_pubmed` | `false` | Disable PubMed retrieval |
| `--use_direct_openfda` | `false` | Use direct HTTP client (no MCP) |
| `--openai_model` | `gpt-4.1-mini` | OpenAI model |
| `--temperature` | `0.2` | LLM temperature |
| `--max_tokens` | `900` | LLM max tokens |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `NCBI_EMAIL` | Recommended | Email for NCBI Entrez API (PubMed) |
| `NCBI_API_KEY` | No | NCBI API key (increases rate limit) |
| `OPENFDA_API_KEY` | No | openFDA API key |
| `CHEMBOT_DEBUG` | No | Set to `1` for MCP debug logging |
| `OPENFDA_MCP_COMMAND` | No | Custom MCP server command (default: `python`) |
| `OPENFDA_MCP_ARGS` | No | Custom MCP server args (default: `mcp_servers/openfda_server.py`) |
| `PUBMED_MCP_COMMAND` | No | Custom PubMed MCP server command |
| `PUBMED_MCP_ARGS` | No | Custom PubMed MCP server args |

### Custom MCP Server Invocation

By default, clients launch local MCP servers using `python mcp_servers/<server>.py`. You can customize via environment variables listed above.

For standalone debug mode:
```bash
fastmcp dev mcp_servers/openfda_server.py
fastmcp dev mcp_servers/pubmed_server.py
```