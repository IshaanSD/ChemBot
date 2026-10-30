# ChemBot

ChemBot is a CLI biomedical Q&A agent built with:
- **LangGraph** (graph-based agent control loop with multi-route fan-out)
- **Model Context Protocol (MCP)** (server/client architecture wrapping openFDA, PubMed, and RxNorm)
- **ChromaDB** (persistent local vector store)
- **Sentence Transformers** (local embeddings)
- **OpenAI** (LLM responses)

ChemBot answers using ONLY retrieved evidence from three sources:
- **FDA Drug Labels** — official regulatory prescribing information
- **FAERS** — FDA Adverse Event Reporting System data
- **PubMed** — published biomedical literature via NCBI Entrez

A **query planner** normalizes drug names via RxNorm (brand → generic ingredient,
misspelling-tolerant) before retrieval, and every answer ends with a **verified Sources**
section built in code from the retrieved metadata — so each `(Excerpt N)` citation traces
to a real FDA label section, FAERS report, or PubMed article (with PMID + link).

---

## Architecture

```
User
  │
  ▼
Query Planner (drug normalization) ──► RxNorm MCP Server
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
Answer Generation (OpenAI) + verified Sources section
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

All three MCP servers use **stdio transport** — ChemBot automatically spawns and manages them as subprocesses:

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
| RxNorm | `mcp_servers/rxnorm_server.py` | RxNav / RxNorm | `search_drug`, `lookup_generic_name`, `lookup_brand_name`, `normalize_drug_name` |

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

## Evaluation

`tests/eval/chembot_eval.py` runs ChemBot end-to-end on a dataset and scores three
safety-critical properties — **groundedness / faithfulness**, **citation verification**,
and **refusal / false-premise handling**:

- Each answer sentence is split into **atomic claims**, and each is judged against the
  excerpts the sentence **cites** (so dense bullets get partial credit, not all-or-nothing).
- The support judge is pluggable; the default is the open-weight **MiniCheck** grounding
  checker (`nli` and `dummy` backends also available) — answer generation uses OpenAI, but
  **grading uses no API**.
- Out-of-evidence and adversarial cases must refuse — or refute a false premise from the
  evidence — without affirming it.

```bash
# default: MiniCheck (roberta-large, CPU)
PYTHONPATH=src python tests/eval/chembot_eval.py --judge minicheck

# stronger CPU checker (2048-token window)
PYTHONPATH=src python tests/eval/chembot_eval.py --judge minicheck --minicheck_model deberta-v3-large

# cheap plumbing check (no grading model loaded)
PYTHONPATH=src python tests/eval/chembot_eval.py --judge dummy --limit 2
```

MiniCheck is an optional grading dependency:
```bash
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git" accelerate sentencepiece
```

Results are printed as a summary (faithfulness, citation validity/support, refusal rate) and
written to `eval_report.json`. The dataset lives in `tests/eval/datasets/chembot-eval.json`.

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

To inspect a server standalone (list tools, call them headlessly — no Node required):
```bash
fastmcp inspect mcp_servers/openfda_server.py
fastmcp call mcp_servers/pubmed_server.py search_pubmed query="semaglutide" max_results=2 --json
```

For the interactive web Inspector (requires Node/`npx`):
```bash
mcp dev mcp_servers/openfda_server.py
```

> The older `fastmcp dev <file>` syntax no longer works on FastMCP 3.x, where `dev`
> became a command group. Use `fastmcp inspect` / `fastmcp call`, or `mcp dev`.