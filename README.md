# SkinSight

**Author:** Prerna

SkinSight is an educational prototype that classifies a skin-lesion photograph with a local deep-learning model, then uses a language model to explain the top predictions.

This is **not** a diagnostic product. Predictions and generated text can be wrong. A qualified clinician must interpret any real medical concern.

## What it does

1. Loads **DenseNet121 (Focal Loss v2)** from Hugging Face and runs inference on this machine.
2. Accepts a JPG, PNG, or WEBP upload in the browser.
3. Shows the **top three** class scores as percentage bars.
4. Asks OpenRouter for a structured information card for each of those three labels.
5. Offers a short follow-up chat grounded in those predictions.
6. Includes a separate notebook that **evaluates six** published checkpoints on a public test set.

Classification never sends the image to OpenRouter. Only predicted names and your questions go to the language-model API.

## Layout

```
SkinSight/
  app/                 FastAPI app, config, inference, LLM client
  web/index.html       Browser UI (no build step)
  notebooks/           Test-set evaluation of six models
  results/             CSV and figures written by the notebook
  samples/             Optional local demo images
  run.py               Starts the web server
  requirements.txt
  .env.example
```

## Requirements

- Python 3.9 or newer
- An OpenRouter API key for cards and chat (classification works without it)
- Internet on first run so Hugging Face can download model weights (~28 MB for the default checkpoint)
- A GPU is optional. CPU inference is slower but supported.

## Setup

```bash
cd SkinSight
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
copy .env.example .env
```

The extra `torch` line is for CPU-only Windows. Skip it if you already have a working CUDA PyTorch install.

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY`. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys).

If you skip the key, `/api/classify` still works. `/api/card` and `/api/chat` return a configuration error until the key is present.

## Start the application

From the `SkinSight` folder (the directory that contains `run.py`):

```bash
python run.py
```

Open [http://localhost:8001](http://localhost:8001).

The first start downloads the classifier weights and can take about a minute. Later starts reuse the local Hugging Face cache.

## Using the web app

1. Wait until the setup panel says the server is ready.
2. Drop or select a lesion image.
3. Click **Analyze image**.
4. Read the top-three bars, then the three information cards.
5. Use the chat box for follow-up questions.

Supported uploads: JPEG, PNG, WEBP.

## Classifier

| Setting | Default |
|---|---|
| Architecture | DenseNet121 fine-tuned with focal loss (v2) |
| Display name | DenseNet121 (Focal Loss v2) |
| Preprocessing | That checkpoint's `AutoImageProcessor` (same path as the evaluation notebook) |
| Device | CUDA GPU 0 if available, otherwise CPU |

The Hub repository used at load time is set in `app/config.py` (`HF_MODEL_ID`). Override it in `.env` only if you intentionally switch checkpoints.

The label list comes from the model config (`id2label`). On the evaluation dataset those names include melanoma, melanocytic nevi, basal cell carcinoma, monkeypox, and other lesion types — fourteen classes in total.

## Language models

Cards and chat call [OpenRouter](https://openrouter.ai/). The default primary model is a free DeepSeek chat model. If that model returns HTTP 404 or 429, SkinSight tries a list of other free models defined in `app/config.py`.

The key is read from the environment on the server. It is not embedded in the HTML page.

## Evaluation notebook

`notebooks/evaluate_classifiers.ipynb` compares six Hugging Face checkpoints on:

- Dataset: `ahmed-ai/skin-lesions-classification-dataset`
- Split: `test` (3,674 images in the reference run)

Models (same six as the reference evaluation):

1. DenseNet121 — Focal Loss v2
2. DenseNet121
3. DenseNet121 — Focal Loss
4. ResNet18
5. ResNet50
6. MobileNetV2

Checkpoint IDs used for loading are defined in `app/config.py`.

After a completed run, `results/` should contain:

- `model_comparison_summary.csv`
- `model_comparison_bar.png`
- `confusion_matrices_all.png`
- `per_class_f1_heatmap.png`

Run the notebook from Jupyter with the SkinSight virtual environment selected. A full six-model pass is slow on CPU. Do not type metrics into the CSV by hand; only keep numbers the notebook writes.

A previous evaluation of this same dataset and these same checkpoints (stored in the original research notebook, not re-run here) ranked **DenseNet121-focal-loss-v2** highest on weighted F1. Treat that as historical context until you generate a fresh `results/` folder.

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Server-side LLM key | empty |
| `SKINSIGHT_HOST` | Bind address | `0.0.0.0` |
| `SKINSIGHT_PORT` | HTTP port | `8001` |
| `HF_MODEL_ID` | Live classifier Hub path | DenseNet121 (Focal Loss v2), set in `app/config.py` |
| `OPENROUTER_PRIMARY_MODEL` | First LLM to try | `deepseek/deepseek-v4-flash:free` |
| `EVAL_DATASET_ID` | Notebook dataset | `ahmed-ai/skin-lesions-classification-dataset` |
| `EVAL_SPLIT` | Notebook split | `test` |
| `EVAL_BATCH_SIZE` | Notebook batch size | `32` |
| `SKINSIGHT_RESULTS_DIR` | Output folder name | `results` |
| `SKINSIGHT_SKIP_MODEL_LOAD` | Skip Hub download at startup (`1` for smoke tests only) | unset |

Paths in `app/config.py` are resolved from the project root, so you can start the server from that folder without extra `PYTHONPATH` setup when using `python run.py`.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | Model, device, whether an LLM key is set |
| `POST` | `/api/classify` | Multipart field `image` → ranked scores |
| `POST` | `/api/card` | JSON `{ "label", "confidence" }` → educational card |
| `POST` | `/api/chat` | JSON `{ "system_prompt", "history" }` → reply |

## Stack

- FastAPI and Uvicorn
- PyTorch and Hugging Face Transformers
- Pillow
- OpenRouter (chat completions)
- Datasets, scikit-learn, matplotlib, and seaborn for evaluation

## License and use

SkinSight is a project by **Prerna**. Use this repository for study and demonstration. Generated medical text is produced by an LLM and can be incomplete or incorrect. Do not rely on SkinSight for clinical decisions.
