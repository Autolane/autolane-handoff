# Data Preparation & Annotation

Tools for preparing training datasets in Llava format for Cosmos Reason 2 SFT.

## Pipeline

1. `bootstrap_annotations.py` — Run Cosmos Reason 2 zero-shot on annotated clips to generate draft QA pairs
2. Human review + refinement
3. `prepare_dataset.py` — Convert to Llava format JSON for SFT

## Target

- 200-250 synthetic clips
- 500-700 QA pairs covering safety, OCR, compliance, and handoff planning
