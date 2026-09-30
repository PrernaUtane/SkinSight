# SkinSight — interview questions

**Author:** Prerna

1. **Who built SkinSight?**  
   Prerna. The app is an independent educational prototype. Published CNN checkpoints are loaded for inference; they are not presented as the project author.

2. **What problem does SkinSight address?**  
   Multi-class identification of skin-lesion photographs for education, not diagnosis.

2. **Why not train from scratch?**  
   Lesion datasets are small relative to ImageNet. Transfer learning from pretrained CNNs is the practical approach; this repo only evaluates and serves published checkpoints.

3. **Why DenseNet121 as the live model?**  
   Feature reuse with relatively few parameters. The focal-loss v2 Hub checkpoint was the strongest of the six models in the reference evaluation.

4. **What is focal loss and why use it?**  
   It down-weights easy majority examples so the model focuses on hard or rare classes. Skin datasets are typically imbalanced.

5. **How does inference preprocessing stay consistent with evaluation?**  
   Both use that checkpoint’s `AutoImageProcessor` instead of a hardcoded torchvision pipeline.

6. **What happens at `/api/classify`?**  
   Multipart upload → type check (JPG/PNG/WEBP) → Pillow RGB image → processor → logits → softmax → ranked `{label, score}`.

7. **Why return all classes if the UI shows top 3?**  
   The API stays complete for debugging and notebooks; the frontend slices the first three.

8. **Where do class names come from?**  
   `model.config.id2label` on the Hugging Face classifier, not a hardcoded list in the web app.

9. **How is the LLM keyed?**  
   `OPENROUTER_API_KEY` is read server-side from `.env`. The browser never sees the key.

10. **What if the OpenRouter key is missing?**  
    Classification still works. `/api/card` and `/api/chat` return HTTP 503 with a configuration message. The status endpoint sets `llm_configured: false`.

11. **How do fallback models work?**  
    Completions try the primary model, then a list of free models. HTTP 404/429 triggers the next model.

12. **Does chat send the image to the LLM?**  
    No. Only predicted labels, confidences, and the user’s text.

13. **Which dataset and split are used for metrics?**  
    `ahmed-ai/skin-lesions-classification-dataset`, **test** split, 14 classes.

14. **Why accuracy plus F1?**  
    Accuracy can hide minority-class failure. Weighted/macro F1 and the classification report show per-class behavior.

15. **What is a confusion matrix telling you here?**  
    Which true classes are predicted as which others (for example melanoma vs nevi). Normalized rows make class size less misleading.

16. **Why ROC-AUC in a 14-class setting?**  
    Weighted one-vs-rest AUC summarizes ranking quality across classes when probabilities are well defined.

17. **ResNet18 vs ResNet50 vs MobileNetV2?**  
    Capacity vs efficiency. In the reference run, ResNet50 and MobileNetV2 lagged DenseNet variants; MobileNetV2 was weakest. Re-run the notebook for current numbers.

18. **CPU vs GPU?**  
    The app uses CUDA device 0 if available, else CPU. Serving one image on CPU is acceptable; six-model test-set eval is much slower on CPU.

19. **Main failure modes you would mention?**  
    Domain shift (phone photos vs training images), similar-looking lesions, LLM hallucination in cards, no clinical validation.

20. **How would you extend the project?**  
    Calibration, dermoscopic-only vs clinical-photo splits, Grad-CAM, a second independent test set, and a human-in-the-loop disclaimer UX — not more training in the serving repo unless data and labels are audited.
