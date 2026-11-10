# LLM Lab: Fine-Tune, Quantize, Ship

A self-guided assignment. You'll teach a small coding model a narrow skill (text-to-SQL),
measure whether it worked, then compress it in two formats (MLX and GGUF) and measure
what compression cost you. At the end your own model runs in LM Studio, Ollama and VS Code.

This runbook tells you **what** to do and **what to think about**. It deliberately doesn't
give you the code or exact flags for the core steps; finding them in `--help` output and
the docs is part of the exercise. The commands that are spelled out are plumbing, not the
lesson.

**Hardware budget:** M5 Pro, 24 GB unified memory. Everything here fits; Lab 3 is the tight one.

---

## Setup (once)

```bash
cd ~/repos/hands-on-ml-notes/llm-lab
./setup.sh
```

This installs the Python env (`uv`, Python 3.12) and downloads everything you need:

| What | Where |
|---|---|
| Base model `Qwen/Qwen2.5-Coder-1.5B-Instruct` (bf16 safetensors) | `~/models/hf/Qwen2.5-Coder-1.5B-Instruct` |
| Task data `b-mc2/sql-create-context` (~78k question / schema / SQL triples) | `data/raw/sql-create-context/` |
| Calibration + perplexity text `wikitext-2-raw-v1` | `data/raw/wikitext/` |
| llama.cpp converter scripts, pinned to your Homebrew build | `vendor/llama.cpp/` |

Tools you'll use:

- **MLX side:** `uv run mlx_lm.generate`, `mlx_lm.lora`, `mlx_lm.fuse`, `mlx_lm.convert`, `mlx_lm.server`
- **GGUF side:** `vendor/llama.cpp/convert_hf_to_gguf.py`, `llama-quantize`, `llama-imatrix`, `llama-perplexity`, `llama-bench`, `llama-server`
- **Runtimes:** LM Studio (`~/.lmstudio/bin/lms`), Ollama, Continue in VS Code

Every CLI has `--help`. Read them; they're short.

Keep your work in this folder: scripts in `scripts/`, notes and results in `results.md`.
`data/`, `adapters/` and `runs/` are gitignored, so large files stay out of git.

---

## Lab 0: Orientation and a speed baseline (~30 min)

Goal: get comfortable with the runtimes, and answer "is MLX faster than llama.cpp on my Mac?"
with your own numbers.

1. Generate a completion from the base model with `mlx_lm.generate`. Note the prompt and
   generation tokens/sec it reports.
2. Load `qwen2.5-coder-7b` in LM Studio twice, once with the GGUF already linked from
   Ollama and once with an MLX build downloaded from LM Studio's model browser. Compare
   tokens/sec on the same prompt.
3. Run `llama-bench` on the Ollama-linked GGUF (`~/models/ollama/...`).

**Write down in `results.md`:**
- A table: runtime × format × prompt-processing tok/s × generation tok/s.
- Why might prompt processing and generation favour different backends?
  (Hint: one is compute-bound, the other memory-bandwidth-bound. Which is which?)

---

## Lab 1: Data (~1–2 h)

Goal: turn raw data into training files and, more importantly, understand them.

1. Load the raw dataset (the `datasets` library or pandas) and look at 20 random rows by hand.
   What are the three fields? What does a typical schema look like? How long are things?
2. Plot token-length distributions (use the base model's tokenizer, not character counts).
   Pick a `max_seq_length` and justify it.
3. Design your prompt. Decide what goes in the system / user / assistant turns and how the
   schema is presented. Use the model's **chat template**; don't hand-roll special tokens.
4. Write `scripts/prepare_data.py` that produces `data/sql/{train,valid,test}.jsonl` in a format
   `mlx_lm.lora` accepts. (Find the supported formats in the mlx-lm LoRA docs. There are
   several; pick one and know why.)

**Decisions to justify in `results.md`:**
- How big is each split? Do you need all 78k rows to learn this skill on a 1.5B model?
- Could near-duplicate questions or identical schemas leak between train and test? Check.
- Seed everything. Can someone re-run your script and get byte-identical files?

**Checkpoint:** `head -c 600 data/sql/train.jsonl` looks like what you intended, and every
split parses as valid JSONL.

---

## Lab 2: Baseline eval before training anything (~1–2 h)

Goal: build the yardstick before you build the thing. An eval you write *after* training
is an eval you'll unconsciously tune toward the result you wanted.

Write `scripts/eval.py` that takes a model (and optionally an adapter), runs it on N test
examples with greedy decoding, and reports:

1. **Exact match:** normalized string equality with the gold SQL. Decide what "normalized" means.
2. **Execution accuracy:** create the schema in an in-memory `sqlite3` DB, run gold and
   predicted SQL, compare results.
3. **Valid-SQL rate:** fraction of predictions that execute at all.

> **Think hard about #2.** The schemas in this dataset come with no rows. What does "same
> result" mean on empty tables, and how could a wrong query score as correct? Fix your
> metric before you trust it.

Run it on the base model (N≈200 is enough to start). Also run it on `qwen2.5-coder-7b`
zero-shot, through LM Studio's or Ollama's OpenAI-compatible API, for an "is fine-tuning
even worth it?" reference point.

**Write down:** baseline numbers, plus 5 failure cases with your diagnosis of each.
Are the failures about SQL knowledge, or about formatting (extra prose, markdown fences)?

---

## Lab 3: LoRA fine-tune (~2–3 h, most of it waiting)

Goal: train an adapter with `mlx_lm.lora` and understand every knob you turn.

Before running anything, answer from the docs and your Part II notes:
- What matrices does LoRA add, and to which layers? What do `rank` and `scale` (alpha) do?
- What does masking the prompt in the loss change? Would you want it here?
- Roughly how many trainable parameters will you have vs. 1.5B total?

Then train:
1. Start small: a few hundred iterations, a modest batch size, and evaluate every N steps.
   Save adapters to `adapters/run-01/`.
2. Watch train and val loss. Plot both from the log; you've done this before in the PyTorch chapter.
3. Run `scripts/eval.py` with the adapter. Compare to the Lab 2 baseline.

**Experiments (pick at least two, change one variable at a time):**
- Rank: 4 vs 16 vs 64
- Number of layers adapted: last few vs all
- Learning rate: ×0.3 / ×3
- Dataset size: 1k vs 5k vs 20k examples (a learning curve)
- LoRA vs DoRA (`--fine-tune-type`)
- **QLoRA:** train the adapter on a 4-bit base. What changes in memory and in accuracy?

**If you run out of memory:** look for gradient checkpointing, smaller batch, shorter
`max_seq_length`, fewer adapted layers. Watch memory pressure in Activity Monitor.

**Checkpoint:** a run that clearly beats the base model on execution accuracy, plus a
loss curve showing whether it's still improving or overfitting.

---

## Lab 4: MLX quantization (~1 h)

Goal: see what quantization costs, in MLX first because it's the simplest path.

1. Fuse your best adapter into the base weights (`mlx_lm.fuse`) → `~/models/local/sql-coder-1.5b-mlx-bf16/`.
2. Quantize the fused model with `mlx_lm.convert` at 8, 4 and 3 bits, and try two group sizes.
3. Eval every variant with your Lab 2 script. Also record size on disk and generation tok/s.

**Questions:**
- What is a quantization "group", and why does group size trade size for accuracy?
- Did the fine-tuned skill degrade faster or slower than general ability as you dropped bits?
  (How would you even measure "general ability"? See Lab 5.)

---

## Lab 5: GGUF quantization with llama.cpp (~2 h)

Goal: the same model in the format Ollama and most of the ecosystem use, with the more
sophisticated k-quants and importance matrices.

1. Convert the **fused bf16** model (not the MLX-quantized one; why not?) to an f16 or bf16
   GGUF with `uv run python vendor/llama.cpp/convert_hf_to_gguf.py`.
2. Make plain quants with `llama-quantize`: at least `Q8_0`, `Q4_K_M`, `Q3_K_M`, `Q2_K`.
3. Turn the wikitext test split into a plain-text file, and measure **perplexity** for each
   quant with `llama-perplexity`. That's your "general ability" proxy.
4. Build an **importance matrix** with `llama-imatrix`. Choose your calibration text: wikitext,
   your SQL training data, or a mix. Re-quantize `Q3_K_M` and `Q2_K` with it, then re-measure.
5. Run your task eval against each GGUF through `llama-server`'s OpenAI-compatible endpoint.

**Deliverable:** one table and one plot in `results.md`.
Columns: format, bits/weight, size (GB), tok/s, perplexity, exact match, execution accuracy.
Plot size vs. execution accuracy and mark the Pareto frontier.

**Questions:**
- Did imatrix help more at 2-bit or 4-bit? Why would that be?
- Did calibrating on SQL data vs. wikitext change task accuracy vs. perplexity differently?
  What does that tell you about calibration data choice?
- Q4_K_M vs MLX 4-bit at similar size: which wins on your eval, and on speed?

---

## Lab 6: Ship it (~30 min)

1. Put your best GGUF and best MLX quant in `~/models/local/<name>/`. Load both in LM Studio.
2. Import the GGUF into Ollama with a `Modelfile`. You'll need a `TEMPLATE` that matches
   the chat format you trained on. What happens if you get it wrong? Try it and see.
   (Remember: `ollama create` copies the file.)
3. Add it as a chat model in `~/.continue/config.yaml` and ask it for a query from VS Code.

---

## Stretch goals

- Full fine-tune vs LoRA on `Qwen2.5-Coder-0.5B`: memory, time, accuracy.
- Catastrophic forgetting: does your adapter make the model worse at Python? Build a tiny
  Python eval and compare base vs. fine-tuned.
- Swap in a harder dataset (e.g. Spider) and see how far a 1.5B model gets.
- Run the same LoRA recipe with PyTorch and PEFT on your work GPUs; compare against MLX.

---

## Troubleshooting

- **`convert_hf_to_gguf.py` fails on import or an unknown key:** the converter pins a
  `transformers` version (see `vendor/llama.cpp/requirements/`). Try
  `uv run --with 'transformers==<that version>' python vendor/llama.cpp/convert_hf_to_gguf.py ...`.
- **LM Studio doesn't see a new model:** it needs `<publisher>/<model>/<file>` under
  `~/models`. See `~/models/README.md`.
- **New Ollama pull isn't in LM Studio:** `python3 scripts/link-ollama-models.py`.
