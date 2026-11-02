#!/usr/bin/env bash
# Deploy the ChemBot API to Google Cloud Run.
#
# Prereqs:
#   - gcloud CLI authenticated: gcloud auth login && gcloud config set project <PROJECT>
#   - Secret Manager secret 'openai-api-key' created with your OpenAI key:
#       printf '%s' "$OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=-
#   - Cloud Run + Cloud Build APIs enabled.
#
# Key design choices baked in below:
#   --max-instances 1   the daily caps live in process memory, so a single
#                       instance makes them a true global ceiling (more than one
#                       instance would multiply the cap, and the cost with it).
#   MIN_INSTANCES       1 keeps a warm instance (models loaded, cap persists)
#                       with a standing cost; 0 scales to zero (near-zero idle
#                       cost, but cold starts reload models and reset the cap).
#   --concurrency 4     matches MAX_CONCURRENCY; queries are heavy.
#
# Tunables (env): SERVICE, REGION, MIN_INSTANCES (default 1), MEMORY, CPU,
# DAILY_*_LIMIT, RATE_LIMITS, PREWARM_EXAMPLES.
set -euo pipefail

SERVICE="${SERVICE:-chembot-api}"
REGION="${REGION:-us-central1}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
NCBI_EMAIL="${NCBI_EMAIL:?set NCBI_EMAIL (required for PubMed/Entrez)}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:?set ALLOWED_ORIGINS to your web UI origin, e.g. https://USER.github.io}"

# With a warm instance, keep CPU always allocated so it isn't throttled between
# requests; when scaling to zero, use default throttling to minimize idle cost.
if [ "$MIN_INSTANCES" = "0" ]; then
  CPU_FLAG="--cpu-throttling"
  PREWARM_DEFAULT="0"   # avoid running example queries on every cold start
else
  CPU_FLAG="--no-cpu-throttling"
  PREWARM_DEFAULT="1"
fi

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --max-instances 1 \
  --min-instances "$MIN_INSTANCES" \
  --concurrency 4 \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --timeout 120 \
  "$CPU_FLAG" \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest" \
  --set-env-vars "NCBI_EMAIL=${NCBI_EMAIL}" \
  --set-env-vars "ALLOWED_ORIGINS=${ALLOWED_ORIGINS}" \
  --set-env-vars "DAILY_QUERY_LIMIT=${DAILY_QUERY_LIMIT:-50}" \
  --set-env-vars "DAILY_OPENAI_LIMIT=${DAILY_OPENAI_LIMIT:-50}" \
  --set-env-vars "RATE_LIMITS=${RATE_LIMITS:-5/minute;20/hour}" \
  --set-env-vars "MAX_CONCURRENCY=4" \
  --set-env-vars "PREWARM_EXAMPLES=${PREWARM_EXAMPLES:-$PREWARM_DEFAULT}"

echo
echo "Deployed. Service URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
