#!/usr/bin/env bash
# One-time setup for the LLM lab: Python deps, base model, datasets, sanity checks.
# Safe to re-run; downloads are skipped when already present.
set -euo pipefail
cd "$(dirname "$0")"

MODELS="$HOME/models"
BASE_REPO="Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_DIR="$MODELS/hf/Qwen2.5-Coder-1.5B-Instruct"

echo "==> Python env (uv, 3.12)"
uv sync --python 3.12

echo "==> llama.cpp converter source (pinned to the Homebrew build's commit)"
LLAMA_COMMIT=$(llama-cli --version 2>&1 | sed -n 's/.*commit \([0-9a-f]*\).*/\1/p')
if [ ! -d vendor/llama.cpp ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp vendor/llama.cpp
fi
git -C vendor/llama.cpp checkout -q "$LLAMA_COMMIT"

echo "==> Base model: $BASE_REPO -> $BASE_DIR"
uv run hf download "$BASE_REPO" --local-dir "$BASE_DIR"

echo "==> Task dataset: b-mc2/sql-create-context -> data/raw/"
uv run hf download b-mc2/sql-create-context --repo-type dataset \
  --local-dir data/raw/sql-create-context

echo "==> Calibration / perplexity text: wikitext-2-raw-v1 -> data/raw/"
uv run hf download Salesforce/wikitext --repo-type dataset \
  --include "wikitext-2-raw-v1/*" --local-dir data/raw/wikitext

echo "==> Refresh Ollama -> LM Studio links"
python3 scripts/link-ollama-models.py

echo "==> Sanity checks"
uv run python - <<'EOF'
import mlx.core as mx, mlx_lm, transformers, torch, gguf, datasets
print(f"mlx {mx.__version__} (metal={mx.metal.is_available()}) | mlx-lm {mlx_lm.__version__}"
      f" | transformers {transformers.__version__} | torch {torch.__version__}")
EOF
for bin in llama-quantize llama-imatrix llama-perplexity llama-bench llama-server; do
  command -v "$bin" >/dev/null && echo "ok  $bin" || echo "MISSING $bin (brew install llama.cpp)"
done
uv run python vendor/llama.cpp/convert_hf_to_gguf.py --help >/dev/null && echo "ok  convert_hf_to_gguf.py"

echo
echo "Setup complete. Start with Lab 0 in README.md."
