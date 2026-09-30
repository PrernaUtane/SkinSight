"""
Runtime configuration for SkinSight.

Values can be changed with environment variables or a local .env file.
Paths are resolved relative to the project root so the app works
no matter which directory you launch it from.
"""

from pathlib import Path
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Project root: SkinSight/  (parent of the app/ package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


PROJECT_AUTHOR = "Prerna"

# --- Classification model ---
HF_MODEL_ID = env("HF_MODEL_ID", "PrernaUtane/skin-lesion-densenet121")
HF_MODEL_LABEL = env("HF_MODEL_LABEL", "DenseNet121 (Focal Loss v2)")

# --- HTTP server ---
HOST = env("SKINSIGHT_HOST", "0.0.0.0")
PORT = int(env("SKINSIGHT_PORT", "8001"))

# --- Paths ---
WEB_DIR = PROJECT_ROOT / env("SKINSIGHT_WEB_DIR", "web")
INDEX_HTML = WEB_DIR / "index.html"
RESULTS_DIR = PROJECT_ROOT / env("SKINSIGHT_RESULTS_DIR", "results")
SAMPLES_DIR = PROJECT_ROOT / env("SKINSIGHT_SAMPLES_DIR", "samples")

# --- OpenRouter ---
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY", "")
OPENROUTER_URL = env("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_PRIMARY_MODEL = env(
    "OPENROUTER_PRIMARY_MODEL",
    "deepseek/deepseek-v4-flash:free",
)

# Tried in order after the primary model returns 404 or 429.
OPENROUTER_FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-coder:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

OPENROUTER_HTTP_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": f"http://localhost:{PORT}",
    "X-Title": "SkinSight",
}

# Models compared in notebooks/evaluate_classifiers.ipynb (order matches the reference list)
EVAL_MODEL_IDS = [
    "PrernaUtane/DenseNet121-focal-loss-v2",
    "PrernaUtane/skin-lesion-densenet121",
    "PrernaUtane/DenseNet121-focal-loss",
    "PrernaUtane/skin-lesion-resnet18",
    "PrernaUtane/skin-lesion-resnet50",
    "PrernaUtane/skin-lesion-mobileNetv2",
]

EVAL_MODEL_NAMES = [
    "DenseNet121 — Focal Loss v2",
    "DenseNet121",
    "DenseNet121 — Focal Loss",
    "ResNet18",
    "ResNet50",
    "MobileNetV2",
]

EVAL_DATASET_ID = env(
    "EVAL_DATASET_ID",
    "ahmed-ai/skin-lesions-classification-dataset",
)
EVAL_SPLIT = env("EVAL_SPLIT", "test")
EVAL_BATCH_SIZE = int(env("EVAL_BATCH_SIZE", "32"))
