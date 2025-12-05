\# Pneumonia AI — Multimodal Detection (MVP)



This project implements a multimodal pneumonia detection system using:



\- Chest X-ray images

\- Clinical text (symptoms/history)

\- Pretrained models (DenseNet, ClinicalBERT)

\- Fusion model (image + text → final prediction)

\- Explainability (Grad-CAM for CXR, SHAP/LIME for text)



\## Project Reference Specification

The detailed design/spec is stored in:

`/mnt/data/algorithms-pnemonias 12.pdf`



\## Folder Structure

data/

&nbsp; raw/

&nbsp; processed/

&nbsp; sample/

models/

&nbsp; checkpoints/

&nbsp; exports/

src/

&nbsp; preprocessing/

&nbsp; dataloaders/

&nbsp; training/

&nbsp; inference/

&nbsp; explainability/

&nbsp; utils/

deployment/

&nbsp; django\_api/

&nbsp; docker/

notebooks/

tests/



\## Environment Activation

ai-env\\Scripts\\activate



\## To install requirements

pip install -r requirements.txt



