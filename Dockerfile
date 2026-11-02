# ChemBot API container.
FROM python:3.12-slim

WORKDIR /app

# Python deps (API server + core pipeline). torch/sentence-transformers come in
# as wheels, so no system build toolchain is needed.
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code and the stdio MCP servers (spawned as `python mcp_servers/*.py`).
COPY src/ src/
COPY mcp_servers/ mcp_servers/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    CHEMBOT_CHROMA_DIR=/app/chroma_db \
    HF_HOME=/app/.hf \
    PORT=8080

# Bake the embedding + reranker weights into the image so cold starts don't
# download them and runtime needs no network for model weights.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn chembot.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
