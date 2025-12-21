# ChemBot

ChemBot is a CLI medication-label Q&A agent built with:
- LangGraph (graph-based agent control loop)
- ChromaDB (persistent vector store)
- Sentence Transformers (local embeddings)
- OpenAI (LLM responses)

ChemBot answers using ONLY retrieved FDA drug label excerpts.

We use the openFDA drug products labels API to get information on adverse reactions, warnings, contraindications, drug interactions, and use guidleines for thousands of medications. More information about this API is available [here](https://open.fda.gov/apis/drug/label/how-to-use-the-endpoint/).

## Install ChemBot

```bash
# change directory to ChemBot
pip install -r requirements.txt
export OPENAI_API_KEY="..."
```

## Run ChemBot


```bash
python -m chembot.graph_cli --chroma_dir ./chroma_db
```

To switch OpenAI model:

```bash
python -m chembot.graph_cli \
  --chroma_dir ./chroma_db \
  --openai_model gpt-4.1-mini
```
## (Optional) Use openFDA API key
openFDA (wihtout a key) has a rate limit of 1000.
For higher than 1000 requests a day (up to 120,000 requests per day), make a free openFDA API key, and then use it in ChemBot by running:
```bash
python -m chembot.graph_cli \
  --chroma_dir ./chroma_db \
  --openfda_api_key YOUR_OPENFDA_KEY
```