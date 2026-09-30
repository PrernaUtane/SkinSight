# SkinSight — project explanation

**Author:** Prerna

SkinSight is Prerna’s educational web app for skin-lesion classification. It is not a clinical product.

## Problem

Skin lesions are visually similar across conditions (for example melanoma vs melanocytic nevi). SkinSight is an **educational classifier**: a user uploads a photograph, a convolutional network predicts a lesion class, and a language model explains the top predictions. It is **not** a clinical diagnostic device.

## Dataset

Evaluation uses the Hugging Face dataset `ahmed-ai/skin-lesions-classification-dataset`. Metrics are computed on the **`test` split** (3,674 images in the reference notebook). There are **14 classes**, including melanoma, melanocytic nevi, basal cell carcinoma, squamous cell carcinoma, actinic keratoses, benign keratosis-like lesions, dermatofibroma, vascular lesions, chickenpox, cowpox, HFMD, measles, monkeypox, and healthy skin.

The live web app does **not** load this dataset. It classifies whatever JPEG/PNG/WEBP the user uploads. Class names come from the checkpoint’s `id2label`.

## Preprocessing

The live model and the evaluation notebook both use Hugging Face **`AutoImageProcessor`** for the loaded checkpoint (resize/crop/normalize as stored with that model). Images are converted to RGB first. This avoids a train/serve mismatch from hand-coded ImageNet transforms.

## DenseNet121

The production classifier is **DenseNet121 (Focal Loss v2)**. DenseNet connects each layer to all later layers, which reuses features and keeps the parameter count smaller than a comparably deep ResNet. The head is a 14-way linear classifier over ImageNet-initialized convolutional features. The published Hub path used for loading is stored in `app/config.py` and is not shown in the UI.

## Transfer learning

Checkpoints were fine-tuned on lesion images from ImageNet-pretrained backbones. SkinSight **does not train**. It downloads published weights and runs `eval()` inference.

## Inference

`app/inference.py` loads `AutoImageProcessor` + `AutoModelForImageClassification`, moves tensors to CUDA if available (else CPU), applies softmax to logits, and returns **all classes sorted by probability**. The UI shows the **top 3**. FastAPI validates upload type, opens the file with Pillow, and never sends the image to OpenRouter.

## Six-model evaluation

`notebooks/evaluate_classifiers.ipynb` scores six published Hub models on the same test split, batch size 32:

1. DenseNet121 — Focal Loss v2  
2. DenseNet121  
3. DenseNet121 — Focal Loss  
4. ResNet18  
5. ResNet50  
6. MobileNetV2  

It does not invent numbers. CSVs and figures are written to `results/` only after a real run.

## Metrics

For each model: **accuracy**, **weighted/macro F1**, **weighted precision**, **weighted recall**, a full **classification report**, and **weighted one-vs-rest ROC-AUC** when sklearn can compute it. Rankings use weighted F1.

## Confusion matrix

The notebook plots **row-normalized** confusion matrices (true class on rows, predicted on columns) plus a **per-class F1 heatmap** across models.

## Limitations

- Photographs, lighting, and cropping differ from dermoscopic datasets; scores can be wrong.  
- Class imbalance makes accuracy look better than minority-class recall.  
- Focal-loss variants still confuse similar pigmented lesions.  
- LLM cards/chat are generic educational text, not patient-specific advice.  
- CPU evaluation of six models on thousands of images is slow.  
- No fairness audit, calibration plots, or external hospital validation in this repo.
