#!/bin/bash
# example_run.sh
# Runs ChemBot with a sample input ("Aspirin") followed by "q" (quit) to demonstrate execution and test the system.

# Get the directory of the script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Set PYTHONPATH to the src directory so the chembot module can be found
export PYTHONPATH="src"

# Define the local chroma DB directory
CHROMA_DIR="./chroma_db"

echo "======================================================================"
echo "Starting ChemBot Test Run (1/2) - DIRECT CLIENT FALLBACK"
echo "  Chroma Directory: $CHROMA_DIR"
echo "  Inputs: 'Aspirin', 'q'"
echo "======================================================================"
echo ""

# Run in Direct client mode (default)
printf "Aspirin\nq\n" | python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"

echo ""
echo "======================================================================"
echo "Starting ChemBot Test Run (2/2) - MCP MODE"
echo "  Chroma Directory: $CHROMA_DIR"
echo "  Inputs: 'Aspirin', 'q'"
echo "======================================================================"
echo ""

# Run in MCP client mode (using --use_mcp_openfda)
printf "Aspirin\nq\n" | python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR" --use_mcp_openfda

echo ""
echo "======================================================================"
echo "ChemBot Test Runs Completed Successfully"
echo "======================================================================"
