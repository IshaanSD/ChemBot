# Deploying ChemBot

Two pieces: the **API** (Cloud Run, Dockerized) and the **web UI** (static, GitHub Pages).

## API → Cloud Run

```bash
# one-time: store the OpenAI key as a secret
printf '%s' "$OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=-

# deploy (builds the Dockerfile via Cloud Build)
NCBI_EMAIL="you@example.com" \
ALLOWED_ORIGINS="https://USER.github.io" \
bash deploy/cloud-run.sh
```

### Why a single warm instance
The rate/cost caps (`chembot.usage`) are **in-process counters**. With more than
one instance each would keep its own counter, so the "daily cap" would multiply
by the instance count — and the cost with it. `--max-instances 1` makes the cap a
true global ceiling; `--min-instances 1 --no-cpu-throttling` keeps models warm.
If you ever need to scale out, move the counters to a shared store (Firestore /
Redis) first.

### Cost controls (all env-configurable)
| Env | Default | Effect |
|-----|---------|--------|
| `DAILY_QUERY_LIMIT` | 50 | Master cap: live (non-cached) questions per UTC day |
| `DAILY_OPENAI_LIMIT` | 50 | OpenAI completions/day (backstop) |
| `DAILY_OPENFDA_LIMIT` / `DAILY_PUBMED_LIMIT` / `DAILY_RXNORM_LIMIT` | 150 | Per-provider calls/day |
| `RATE_LIMITS` | `5/minute;20/hour` | Per-IP rate limit |
| `MAX_CONCURRENCY` | 4 | Max simultaneous in-flight questions |
| `PREWARM_EXAMPLES` | 1 (deploy) | Run example questions once at boot so they're free cache hits |

Cache hits never count against any cap, so repeated and example questions stay
free even after the daily limit is reached.

## Web UI → GitHub Pages

Set a repo **variable** `API_BASE_URL` to the Cloud Run service URL, then push to
`main` — the `deploy-web` workflow publishes `web/` to Pages (substituting the
URL into `index.html`). Point `ALLOWED_ORIGINS` on the API at the Pages origin.

Locally you can also open `web/index.html?api=http://localhost:8080`.
