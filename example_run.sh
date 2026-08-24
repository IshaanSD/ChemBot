#!/bin/bash
# example_run.sh
# Runs ChemBot with sample queries exercising RxNorm normalization and multi-source routing.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
export PYTHONPATH="src"
CHROMA_DIR="./chroma_db"
export CHEMBOT_DEBUG=1
export NCBI_EMAIL="${NCBI_EMAIL:-test@chembot.dev}"

echo "======================================================================"
echo "Starting ChemBot RxNorm Integration Tests"
echo "======================================================================"
echo ""

# 1. Brand name -> Generic
echo "Test 1: Brand name (Lipitor)"
printf "What are the side effects of Lipitor?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"
echo ""

# 2. Generic name
echo "Test 2: Generic name (ibuprofen)"
printf "What is the dosage for ibuprofen?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"
echo ""

# 3. Misspellings
echo "Test 3: Misspelled drug name (tylenal)"
printf "What are the contraindications for tylenal?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"
echo ""

# 4. Unknown drug
echo "Test 4: Unknown drug name (MadeUpDrugXYZ123)"
printf "What are the side effects of MadeUpDrugXYZ123?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"
echo ""

# 5. Combination product (Percocet contains acetaminophen and oxycodone)
echo "Test 5: Combination product (Percocet)"
printf "What are the contraindications of Percocet?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"
echo ""

# 6. Ambiguous / Generic term (e.g. "painkiller") 
echo "Test 6: Ambiguous drug class/term (painkiller)"
printf "What are the side effects of a painkiller?\nq\n" | \
  python -u -m chembot.graph_cli --chroma_dir "$CHROMA_DIR"
echo ""

echo "======================================================================"
echo "ChemBot Test Runs Completed Successfully"
echo "======================================================================"
