#!/usr/bin/env python3
"""Symlink Ollama's GGUF blobs into ~/models/ollama/ so LM Studio can load them.

No bytes are copied: Ollama keeps owning the files, LM Studio just sees links.
Re-run after `ollama pull` / `ollama rm`; stale links are pruned.

Where Ollama ships a manifest list (e.g. gpt-oss, qwen3.5) we prefer the
`llamacpp/` variant, which is the stock-llama.cpp-compatible build LM Studio
needs; the default variant targets Ollama's own engine.
"""

import json
import shutil
from pathlib import Path

OLLAMA = Path.home() / ".ollama/models"
LIBRARY = OLLAMA / "manifests/registry.ollama.ai/library"
DEST = Path.home() / "models/ollama"

MODEL = "application/vnd.ollama.image.model"
PROJECTOR = "application/vnd.ollama.image.projector"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"

# Ollama ships these only in its own engine's GGUF layout, with no llamacpp/
# variant; stock llama.cpp (LM Studio) fails with "wrong number of tensors".
OLLAMA_ENGINE_ONLY = {"gemma4"}


def blob(digest: str) -> Path:
    return OLLAMA / "blobs" / digest.replace(":", "-")


def is_gguf(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"GGUF"


def resolve(manifest: Path) -> dict:
    layers = json.loads(manifest.read_text())["layers"]
    # Swap in the llama.cpp-compatible variant when the manifest list has one.
    for layer in layers:
        if layer["mediaType"] == DOCKER_MANIFEST:
            variant = LIBRARY / "llamacpp" / layer["digest"].split(":")[1]
            if variant.exists():
                layers = json.loads(variant.read_text())["layers"]
                break
    return {l["mediaType"]: blob(l["digest"]) for l in layers}


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)  # only ever contains symlinks we created
    for manifest in sorted(LIBRARY.glob("*/*")):
        name = manifest.parent.name
        if name == "llamacpp":
            continue
        slug = f"{name}-{manifest.name}"
        if name in OLLAMA_ENGINE_ONLY:
            print(f"skip  {slug} (Ollama-engine-only; download it in LM Studio)")
            continue
        layers = resolve(manifest)
        model = layers.get(MODEL)
        if not model or not model.exists() or not is_gguf(model):
            print(f"skip  {slug} (no GGUF model layer)")
            continue
        out = DEST / slug
        out.mkdir(parents=True)
        (out / f"{slug}.gguf").symlink_to(model)
        if PROJECTOR in layers:
            # LM Studio pairs vision projectors by the mmproj- prefix.
            (out / f"mmproj-{slug}.gguf").symlink_to(layers[PROJECTOR])
        print(f"link  {slug}")


if __name__ == "__main__":
    main()
