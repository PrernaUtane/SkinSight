# SkinSight

**AI-Powered Skin Lesion Classification & Educational Information Platform**

SkinSight is an educational AI prototype that analyzes skin-lesion images using a fine-tuned **DenseNet121 deep-learning model** and provides structured educational information about the predicted conditions using an LLM.

> **Disclaimer:** SkinSight is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

## Features

* 🧠 **Deep-learning classification** using a fine-tuned DenseNet121 model
* 📷 **Image upload** supporting JPG, PNG, and WEBP
* 📊 **Top-3 predictions** with confidence scores
* 🤖 **LLM-generated educational information** for predicted conditions
* 💬 **Follow-up question chat** grounded in the classification results
* 🔐 **Server-side API key handling** — the OpenRouter API key is never exposed to the browser
* ⚡ **FastAPI backend** with a lightweight browser interface
* 📈 **Model evaluation notebook** comparing six published checkpoints

## How It Works

```text
User uploads skin-lesion image
            │
            ▼
     FastAPI Backend
            │
            ▼
   DenseNet121 Classifier
            │
            ▼
     Top-3 Predictions
            │
            ├───────────────┐
            ▼               ▼
 Confidence Scores     Predicted Labels
                            │
                            ▼
                       OpenRouter
                            │
                            ▼
               Educational Information
                            │
                            ▼
                       Web Interface
```

### Important privacy design

The uploaded image is processed by the local classifier and **is not sent to OpenRouter**.

Only the predicted condition labels, confidence information, and user questions are sent to the language-model API.

## Project Structure

```text
SkinSight/
│
├── app/
│   ├── config.py          # Application configuration
│   ├── inference.py       # Model loading and image inference
│   ├── llm_client.py      # OpenRouter integration
│   ├── main.py            # FastAPI application and endpoints
│   └── schemas.py         # API request/response schemas
│
├── web/
│   └── index.html         # Browser interface
│
├── notebooks/
│   └── evaluate_classifiers.ipynb
│                           # Six-model evaluation
│
├── results/
│   └── README.md          # Evaluation output information
│
├── samples/
│   ├── demo_lesion.jpg
│   └── README.md
│
├── run.py                 # Application entry point
├── requirements.txt       # Python dependencies
├── .env.example           # Environment configuration template
├── PROJECT_EXPLANATION.md # Detailed project explanation
└── INTERVIEW_PREPARATION.md
                            # Interview preparation notes
```

## Tech Stack

| Category          | Technologies                        |
| ----------------- | ----------------------------------- |
| Backend           | FastAPI, Uvicorn                    |
| Deep Learning     | PyTorch, DenseNet121                |
| Model Hub         | Hugging Face                        |
| Image Processing  | Pillow                              |
| LLM               | OpenRouter                          |
| Data & Evaluation | Hugging Face Datasets, scikit-learn |
| Visualization     | Matplotlib, Seaborn                 |
| Frontend          | HTML, CSS, JavaScript               |
| Language          | Python                              |

## Model

The live classifier uses:

**DenseNet121 (Focal Loss v2)**

Hugging Face model:

`PrernaUtane/skin-lesion-densenet121`

The model configuration provides the classification labels through `id2label`.

The application automatically uses:

* **CUDA GPU 0** when available
* **CPU** otherwise

The model weights are downloaded from Hugging Face on the first run and subsequently reused through the local Hugging Face cache.

## Model Evaluation

The repository includes an evaluation notebook comparing six published checkpoints on the test split of:

`ahmed-ai/skin-lesions-classification-dataset`

The six evaluated model variants are:

1. DenseNet121 — Focal Loss v2
2. DenseNet121
3. DenseNet121 — Focal Loss
4. ResNet18
5. ResNet50
6. MobileNetV2

The notebook can generate:

```text
results/
├── model_comparison_summary.csv
├── model_comparison_bar.png
├── confusion_matrices_all.png
└── per_class_f1_heatmap.png
```

Evaluation should be reproduced by running the notebook rather than manually entering metrics.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/PrernaUtane/SkinSight.git
cd SkinSight
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows**

```powershell
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

For CPU-only Windows installations, if required:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Configure environment variables

Create `.env` from the example:

**Windows**

```powershell
copy .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

The API key is loaded by the backend and is not exposed to the frontend.

## Running the Application

Start the FastAPI server:

```bash
python run.py
```

Then open:

```text
http://localhost:8001
```

On startup, the application:

1. Initializes the backend
2. Loads the DenseNet121 classifier
3. Detects the available device
4. Checks whether an OpenRouter API key is configured
5. Starts the web interface

## Using SkinSight

1. Open the application in your browser.
2. Upload a skin-lesion image.
3. Click **Analyze image**.
4. View the top-three predictions and confidence scores.
5. Read the educational information generated for the predictions.
6. Ask follow-up questions using the chat interface.

## API Endpoints

| Method | Endpoint        | Purpose                                    |
| ------ | --------------- | ------------------------------------------ |
| `GET`  | `/`             | Web interface                              |
| `GET`  | `/api/status`   | Application and model status               |
| `POST` | `/api/classify` | Classify uploaded image                    |
| `POST` | `/api/card`     | Generate educational condition information |
| `POST` | `/api/chat`     | Ask follow-up questions                    |

### Example classification request

The `/api/classify` endpoint accepts a multipart form upload:

```text
image=<image file>
```

The response contains the ranked classification results.

## Configuration

Important environment variables include:

| Variable                   | Purpose                            |
| -------------------------- | ---------------------------------- |
| `OPENROUTER_API_KEY`       | Server-side LLM API key            |
| `HF_MODEL_ID`              | Hugging Face classifier repository |
| `SKINSIGHT_HOST`           | Server bind address                |
| `SKINSIGHT_PORT`           | Application port                   |
| `OPENROUTER_PRIMARY_MODEL` | Primary OpenRouter model           |
| `EVAL_DATASET_ID`          | Evaluation dataset                 |
| `EVAL_SPLIT`               | Evaluation dataset split           |
| `EVAL_BATCH_SIZE`          | Evaluation batch size              |

The default live classifier is configured in `app/config.py`.

## Limitations

SkinSight is a prototype and has important limitations:

* Image classification can produce incorrect predictions.
* Confidence scores should not be interpreted as medical certainty.
* LLM-generated information can be incomplete or incorrect.
* The system has not been validated for clinical use.
* Results should not be used to make medical decisions.

For real medical concerns, consult a qualified healthcare professional.

## Project Documentation

Additional documentation is included in the repository:

* `PROJECT_EXPLANATION.md` — detailed technical explanation
* `INTERVIEW_PREPARATION.md` — project-focused interview preparation
* `notebooks/evaluate_classifiers.ipynb` — model evaluation workflow

## Author

**Prerna Utane**

Artificial Intelligence & Data Science

## License and Use

SkinSight is an educational and research project created for learning and demonstration purposes.

It must not be used as a clinical diagnostic tool.
