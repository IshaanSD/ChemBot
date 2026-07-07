#!/bin/bash
# example_run.sh
# Runs ChemBot with sample queries exercising multi-source routing (FDA Label, FAERS, PubMed).
# Tests both direct OpenFDA fallback mode and default MCP mode with debug logging.

# Get the directory of the script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Set PYTHONPATH to the src directory so the chembot module can be found
export PYTHONPATH="src"

# Define the local chroma DB directory
CHROMA_DIR="./chroma_db"

# Enable MCP client debug logging (writes inputs/outputs to stderr)
export CHEMBOT_DEBUG=1

# NCBI requires an email for Entrez API usage
export NCBI_EMAIL="${NCBI_EMAIL:-test@chembot.dev}"

echo "======================================================================"
echo "Starting ChemBot Test Run (1/3) - DIRECT CLIENT FALLBACK (no PubMed)"
echo "  Chroma Directory: $CHROMA_DIR"
echo "  Mode: Direct OpenFDA + PubMed disabled"
echo "  Query: 'Ibuprofen'"
echo "  Expected route: FDA Label only"
echo "======================================================================"
echo ""

# Label-only query with direct client fallback and PubMed disabled
# Note: direct client cannot handle question-format queries (? causes HTTP 400)
printf "Ibuprofen\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR" --use_direct_openfda --disable_pubmed

echo ""
echo "======================================================================"
echo "Starting ChemBot Test Run (2/3) - MCP MODE, MULTI-SOURCE"
echo "  Chroma Directory: $CHROMA_DIR"
echo "  Mode: MCP (Default)"
echo "  Query: 'Is ibuprofen safe during pregnancy?'"
echo "  Expected routes: FDA Label + PubMed (multi-route)"
echo "======================================================================"
echo ""

# Multi-source query: triggers both label (pregnancy marker) and pubmed (pregnancy marker)
printf "Is ibuprofen safe during pregnancy?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"

echo ""
echo "======================================================================"
echo "Starting ChemBot Test Run (3/3) - MCP MODE, PUBMED-ONLY"
echo "  Chroma Directory: $CHROMA_DIR"
echo "  Mode: MCP (Default)"
echo "  Query: 'What recent studies exist on semaglutide and Alzheimer disease?'"
echo "  Expected route: PubMed only"
echo "======================================================================"
echo ""

# PubMed-only query: triggers pubmed (recent, studies markers)
printf "What recent studies exist on semaglutide and Alzheimer disease?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"

echo ""
echo "======================================================================"
echo "ChemBot Test Runs Completed Successfully"
echo "======================================================================"
