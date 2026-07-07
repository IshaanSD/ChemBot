#!/bin/bash
# example_run.sh
# Runs ChemBot with multiple sample queries in debug mode for both direct and MCP execution modes.

# Get the directory of the script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Set PYTHONPATH to the src directory so the chembot module can be found
export PYTHONPATH="src"

# Define the local chroma DB directory
CHROMA_DIR="./chroma_db"

# Enable MCP client debug logging (writes inputs/outputs to stderr)
export CHEMBOT_DEBUG=1

echo "======================================================================"
echo "Starting ChemBot Test Run (1/2) - DIRECT CLIENT FALLBACK"
echo "  Chroma Directory: $CHROMA_DIR"
echo "  Mode: Direct (Fallback)"
echo "  Inputs: 'Ibuprofen' -> 'What are the side effects of ibuprofen?' -> 'q'"
echo "  Verification: (No [MCP CLIENT DEBUG] lines should appear)"
echo "======================================================================"
echo ""

# Feed queries to the CLI agent
# 1. Ask about Ibuprofen (causes label query)
# 2. Ask about side effects of ibuprofen (causes adverse event query)
# 3. Quit
printf "Ibuprofen\nWhat are the side effects of ibuprofen?\nq\n" | python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"

echo ""
echo "======================================================================"
echo "Starting ChemBot Test Run (2/2) - MCP MODE"
echo "  Chroma Directory: $CHROMA_DIR"
echo "  Mode: MCP"
echo "  Inputs: 'Ibuprofen' -> 'What are the side effects of ibuprofen?' -> 'q'"
echo "  Verification: ([MCP CLIENT DEBUG] logs will show in stderr)"
echo "======================================================================"
echo ""

# Feed the same queries in MCP mode
printf "Ibuprofen\nWhat are the side effects of ibuprofen?\nq\n" | python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR" --use_mcp_openfda

echo ""
echo "======================================================================"
echo "ChemBot Test Runs Completed Successfully"
echo "======================================================================"
