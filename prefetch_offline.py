"""Pre-download everything the handson-mlp notebooks fetch at runtime.

Run from inside handson-mlp/ (the notebooks use paths relative to it):

    cd handson-mlp && .venv/bin/python ../prefetch_offline.py [--big]

Hugging Face, torchvision and scikit-learn files go to their default caches
(~/.cache/huggingface, ~/.cache/torch, ~/scikit_learn_data), so they also work
from your own notebooks. Everything else lands in handson-mlp/datasets/ in the
layout the notebooks expect. Safe to re-run: finished items are skipped.
"""

import sys
import tarfile
import traceback
import urllib.request
import zipfile
from pathlib import Path

BIG = "--big" in sys.argv
GH_DATA = "https://github.com/ageron/data/raw/main/"
GH_HOMLP = "https://github.com/ageron/handson-mlp/raw/main/"

failures = []


def step(name):
    def wrap(fn):
        print(f"==> {name}", flush=True)
        try:
            fn()
        except Exception:
            traceback.print_exc()
            failures.append(name)
        return fn
    return wrap


def download(url, path):
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("   ", url, "->", path, flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def extract(archive, dest="datasets"):
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(path=dest, filter="data")


# --- Transformers chapters (14-17) first ---------------------------------

HF_MODELS = [
    # ch14
    "gpt2", "bert-base-uncased", "albert-base-v2",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "huggingface/distilbert-base-uncased-finetuned-mnli",
    # ch15
    "sentence-transformers/all-MiniLM-L6-v2",
    # ch16
    "google/vit-base-patch16-224-in21k",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/dino-vitb8", "openai/clip-vit-base-patch32",
    # ch17
    "facebook/opt-125m", "facebook/opt-350m",
    "google/bigbird-roberta-base", "EleutherAI/gpt-neo-125M",
    # Appendix B / E
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "state-spaces/mamba-790m-hf",
]
HF_BIG_MODELS = [
    "Salesforce/blip2-opt-2.7b",              # ch16, ~15 GB
    "mistralai/Mistral-7B-v0.3",              # ch15, ~14 GB, gated
    "mistralai/Mistral-7B-Instruct-v0.3",     # ch15, ~14 GB, gated
]
# Skip duplicate weight formats so we only keep what transformers loads.
SKIP = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "openvino/*",
        "*.tflite", "coreml/*", "rust_model*", "tf_model*", "flax_model*",
        "consolidated.safetensors"]


@step("Hugging Face datasets (ch14-16)")
def _():
    from datasets import load_dataset
    load_dataset("stanfordnlp/imdb")
    load_dataset(path="ageron/tatoeba_mt_train", name="eng-spa",
                 split=["validation", "test"])
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    load_dataset("tatsu-lab/alpaca", split="train")
    load_dataset("Anthropic/hh-rlhf", split="train")
    load_dataset("timm/oxford-iiit-pet")


@step("Hugging Face models (ch14-17, appendices)")
def _():
    from huggingface_hub import snapshot_download, hf_hub_download
    for repo in HF_MODELS + (HF_BIG_MODELS if BIG else []):
        print("   ", repo, flush=True)
        try:
            snapshot_download(repo, ignore_patterns=SKIP)
        except Exception as e:
            print("    FAILED:", e)
            failures.append(repo)
    hf_hub_download("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "tinyllama-1.1b-chat-v1.0.Q6_K.gguf")


@step("Stable Diffusion Turbo fp16 (ch18)")
def _():
    from huggingface_hub import snapshot_download
    snapshot_download("stabilityai/sd-turbo",
                      allow_patterns=["*.json", "*.txt", "*fp16*"])


@step("Sample images for ch16")
def _():
    download("http://images.cocodataset.org/val2017/000000039769.jpg",
             "datasets/images/two_cats.jpg")
    download("https://homl.info/ladybug", "datasets/images/ladybug.jpg")


# --- Everything else -----------------------------------------------------

@step("Book data files (ch1-3, 13, 14)")
def _():
    for f in ("lifesat.csv", "oecd_bli.csv", "gdp_per_capita.csv"):
        download(GH_DATA + "lifesat/" + f, f"datasets/lifesat/{f}")
    for name in ("housing", "titanic", "ridership"):
        if not Path(f"datasets/{name}").is_dir():
            extract(download(f"{GH_DATA}{name}.tgz", f"datasets/{name}.tgz"))
    if not Path("datasets/jsb_chorales").is_dir():
        extract(download("https://homl.info/bach", "datasets/jsb_chorales.tgz"))
    download("https://homl.info/shakespeare",
             "datasets/shakespeare/shakespeare.txt")
    download(GH_HOMLP + "images/end_to_end_project/california.png",
             "my_california.png")
    download(GH_HOMLP + "images/unsupervised_learning/ladybug.png",
             "my_ladybug.png")
    download(GH_DATA + "ppo_agent_breakout.zip", "my_ppo_agent_breakout.zip")


@step("SpamAssassin corpus (ch3)")
def _():
    root = "http://spamassassin.apache.org/old/publiccorpus/"
    spam_path = Path("datasets/spam")
    for dir_name, tar_name, fname in (("easy_ham", "ham", "20030228_easy_ham"),
                                      ("spam", "spam", "20030228_spam")):
        if not (spam_path / dir_name).is_dir():
            extract(download(f"{root}{fname}.tar.bz2",
                             spam_path / f"{tar_name}.tar.bz2"), spam_path)


@step("scikit-learn datasets (ch3-11, App. C)")
def _():
    from sklearn import datasets as skd
    data_home = Path(skd.get_data_home())
    data_home.mkdir(parents=True, exist_ok=True)
    download(GH_DATA + "olivetti_py3.pkz", data_home / "olivetti_py3.pkz")
    skd.fetch_olivetti_faces()
    skd.fetch_openml("mnist_784", as_frame=False)
    skd.fetch_openml(name="Fashion-MNIST", as_frame=False)
    skd.fetch_california_housing()
    skd.fetch_covtype()


@step("torchvision datasets (ch10-12, 18)")
def _():
    import torchvision
    for cls in (torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST,
                torchvision.datasets.CIFAR10):
        for train in (True, False):
            cls(root="datasets", train=train, download=True)
    for split in ("train", "val", "test"):
        torchvision.datasets.Flowers102(root="datasets", split=split,
                                        download=True)
    if not Path("datasets/hymenoptera_data").is_dir():
        extract(download(
            "https://download.pytorch.org/tutorial/hymenoptera_data.zip",
            "datasets/hymenoptera_data.zip"))


@step("torchvision pretrained weights (ch12)")
def _():
    from torchvision import models
    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)


@step("YOLO weights and sample media (ch12)")
def _():
    from ultralytics import YOLO
    YOLO("yolov9m.pt")
    for f in ("soccer.jpg", "traffic.jpg", "cars.mp4"):
        download(f"https://homl.info/{f}", f"datasets/images/{f}")


@step("Atari ROM check (ch19)")
def _():
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    gym.make("BreakoutNoFrameskip-v4").close()


print()
print("FAILED:", failures) if failures else print("All done.")
sys.exit(1 if failures else 0)
